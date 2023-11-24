# Import libraries
import numpy as np
import pandas as pd
import streamlit as st


import cv2
import skimage
from PIL import Image, ImageColor
from ultralytics import YOLO
from sklearn.metrics import mean_squared_error

import os
import json
import yaml
import time

def get_labels_dics():
    # Get tactical map keypoints positions dictionary
    json_path = "../pitch map labels position.json"
    with open(json_path, 'r') as f:
        keypoints_map_pos = json.load(f)

    # Get football field keypoints numerical to alphabetical mapping
    yaml_path = "../config pitch dataset.yaml"
    with open(yaml_path, 'r') as file:
        classes_names_dic = yaml.safe_load(file)
    classes_names_dic = classes_names_dic['names']

    # Get football field keypoints numerical to alphabetical mapping
    yaml_path = "../config players dataset.yaml"
    with open(yaml_path, 'r') as file:
        labels_dic = yaml.safe_load(file)
    labels_dic = labels_dic['names']
    return keypoints_map_pos, classes_names_dic, labels_dic

def create_colors_info(team1_name, team1_p_color, team1_gk_color, team2_name, team2_p_color, team2_gk_color):
    team1_p_color_rgb = ImageColor.getcolor(team1_p_color, "RGB")
    team1_gk_color_rgb = ImageColor.getcolor(team1_gk_color, "RGB")
    team2_p_color_rgb = ImageColor.getcolor(team2_p_color, "RGB")
    team2_gk_color_rgb = ImageColor.getcolor(team2_gk_color, "RGB")

    colors_dic = {
        team1_name:[team1_p_color_rgb, team1_gk_color_rgb],
        team2_name:[team2_p_color_rgb, team2_gk_color_rgb]
    }
    colors_list = colors_dic[team1_name]+colors_dic[team2_name] # Define color list to be used for detected player team prediction
    color_list_lab = [skimage.color.rgb2lab([i/255 for i in c]) for c in colors_list] # Converting color_list to L*a*b* space
    return colors_dic, color_list_lab

def generate_file_name():
    list_video_files = os.listdir('./outputs/')
    idx = 0
    while True:
        idx +=1
        output_file_name = f'detect_{idx}'
        if output_file_name+'.mp4' not in list_video_files:
            break
    return output_file_name

def detect(cap, stframe, output_file_name, save_output, model_players, model_keypoints,
            hyper_params, ball_track_hyperparams, plot_hyperparams, num_pal_colors, colors_dic, color_list_lab):

    show_k = plot_hyperparams[0]
    show_pal = plot_hyperparams[1]
    show_b = plot_hyperparams[2]
    show_p = plot_hyperparams[3]

    p_conf = hyper_params[0]
    k_conf = hyper_params[1]
    k_d_tol = hyper_params[2]

    nbr_frames_no_ball_thresh = ball_track_hyperparams[0]
    ball_track_dist_thresh = ball_track_hyperparams[1]
    max_track_length = ball_track_hyperparams[2]

    nbr_team_colors = len(list(colors_dic.values())[0])

    if (output_file_name is not None) and (len(output_file_name)==0):
        output_file_name = generate_file_name()

    # Read tactical map image
    tac_map = cv2.imread('../tactical map.jpg')
    tac_width = tac_map.shape[0]
    tac_height = tac_map.shape[1]
    
    # Create output video writer
    if save_output:
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + tac_width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + tac_height
        output = cv2.VideoWriter(f'./outputs/{output_file_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))

    # Create progress bar
    tot_nbr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st_prog_bar = st.progress(0, text='Detection starting.')

    keypoints_map_pos, classes_names_dic, labels_dic = get_labels_dics()

    # Set variable to record the time when we processed last frame 
    prev_frame_time = 0
    # Set variable to record the time at which we processed current frame 
    new_frame_time = 0
    
    # Store the ball track history
    ball_track_history = {'src':[],
                          'dst':[]
    }

    nbr_frames_no_ball = 0

    

    # Loop over input video frames
    for frame_nbr in range(1, tot_nbr_frames+1):

        # Update progress bar
        percent_complete = int(frame_nbr/(tot_nbr_frames)*100)
        st_prog_bar.progress(percent_complete, text=f"Detection in progress ({percent_complete}%)")

        # Read a frame from the video
        success, frame = cap.read()

        # Reset tactical map image for each new frame
        tac_map_copy = tac_map.copy()

        if nbr_frames_no_ball>nbr_frames_no_ball_thresh:
            ball_track_history['dst'] = []
            ball_track_history['src'] = []

        if success:

            #################### Part 1 ####################
            # Object Detection & Coordiante Transofrmation #
            ################################################

            # Run YOLOv8 players inference on the frame
            results_players = model_players(frame, conf=p_conf)
            # Run YOLOv8 field keypoints inference on the frame
            results_keypoints = model_keypoints(frame, conf=k_conf)
            
            

            ## Extract detections information
            bboxes_p = results_players[0].boxes.xyxy.cpu().numpy()                          # Detected players, referees and ball (x,y,x,y) bounding boxes
            bboxes_p_c = results_players[0].boxes.xywh.cpu().numpy()                        # Detected players, referees and ball (x,y,w,h) bounding boxes    
            labels_p = list(results_players[0].boxes.cls.cpu().numpy())                     # Detected players, referees and ball labels list
            confs_p = list(results_players[0].boxes.conf.cpu().numpy())                     # Detected players, referees and ball confidence level
            
            bboxes_k = results_keypoints[0].boxes.xyxy.cpu().numpy()                        # Detected field keypoints (x,y,x,y) bounding boxes
            bboxes_k_c = results_keypoints[0].boxes.xywh.cpu().numpy()                      # Detected field keypoints (x,y,w,h) bounding boxes
            labels_k = list(results_keypoints[0].boxes.cls.cpu().numpy())                   # Detected field keypoints labels list

            

            # Convert detected numerical labels to alphabetical labels
            detected_labels = [classes_names_dic[i] for i in labels_k]

            # Extract detected field keypoints coordiantes on the current frame
            detected_labels_src_pts = np.array([list(np.round(bboxes_k_c[i][:2]).astype(int)) for i in range(bboxes_k_c.shape[0])])

            # Get the detected field keypoints coordinates on the tactical map
            detected_labels_dst_pts = np.array([keypoints_map_pos[i] for i in detected_labels])


            ## Calculate Homography transformation matrix when more than 4 keypoints are detected
            if len(detected_labels) > 3:
                # Always calculate homography matrix on the first frame
                if frame_nbr > 1:
                    # Determine common detected field keypoints between previous and current frames
                    common_labels = set(detected_labels_prev) & set(detected_labels)
                    # When at least 4 common keypoints are detected, determine if they are displaced on average beyond a certain tolerance level
                    if len(common_labels) > 3:
                        common_label_idx_prev = [detected_labels_prev.index(i) for i in common_labels]   # Get labels indexes of common detected keypoints from previous frame
                        common_label_idx_curr = [detected_labels.index(i) for i in common_labels]        # Get labels indexes of common detected keypoints from current frame
                        coor_common_label_prev = detected_labels_src_pts_prev[common_label_idx_prev]     # Get labels coordiantes of common detected keypoints from previous frame
                        coor_common_label_curr = detected_labels_src_pts[common_label_idx_curr]          # Get labels coordiantes of common detected keypoints from current frame
                        coor_error = mean_squared_error(coor_common_label_prev, coor_common_label_curr)  # Calculate error between previous and current common keypoints coordinates
                        update_homography = coor_error > k_d_tol                                         # Check if error surpassed the predefined tolerance level
                    else:
                        update_homography = True                                                         
                else:
                    update_homography = True

                if  update_homography:
                    homog, mask = cv2.findHomography(detected_labels_src_pts,                   # Calculate homography matrix
                                                detected_labels_dst_pts)                  
            if 'homog' in locals():
                detected_labels_prev = detected_labels.copy()                               # Save current detected keypoint labels for next frame
                detected_labels_src_pts_prev = detected_labels_src_pts.copy()               # Save current detected keypoint coordiantes for next frame

                bboxes_p_c_0 = bboxes_p_c[[i==0 for i in labels_p],:]                       # Get bounding boxes information (x,y,w,h) of detected players (label 0)
                bboxes_p_c_2 = bboxes_p_c[[i==2 for i in labels_p],:]                       # Get bounding boxes information (x,y,w,h) of detected ball(s) (label 2)

                # Get coordinates of detected players on frame (x_cencter, y_center+h/2)
                detected_ppos_src_pts = bboxes_p_c_0[:,:2]  + np.array([[0]*bboxes_p_c_0.shape[0], bboxes_p_c_0[:,3]/2]).transpose()
                # Get coordinates of the first detected ball (x_center, y_center)
                detected_ball_src_pos = bboxes_p_c_2[0,:2] if bboxes_p_c_2.shape[0]>0 else None

                if detected_ball_src_pos is None:
                    nbr_frames_no_ball+=1
                else: 
                    nbr_frames_no_ball=0

                # Transform players coordinates from frame plane to tactical map plance using the calculated Homography matrix
                pred_dst_pts = []                                                           # Initialize players tactical map coordiantes list
                for pt in detected_ppos_src_pts:                                            # Loop over players frame coordiantes
                    pt = np.append(np.array(pt), np.array([1]), axis=0)                     # Covert to homogeneous coordiantes
                    dest_point = np.matmul(homog, np.transpose(pt))                              # Apply homography transofrmation
                    dest_point = dest_point/dest_point[2]                                   # Revert to 2D-coordiantes
                    pred_dst_pts.append(list(np.transpose(dest_point)[:2]))                 # Update players tactical map coordiantes list
                pred_dst_pts = np.array(pred_dst_pts)

                # Transform ball coordinates from frame plane to tactical map plance using the calculated Homography matrix
                if detected_ball_src_pos is not None:
                    pt = np.append(np.array(detected_ball_src_pos), np.array([1]), axis=0)
                    dest_point = np.matmul(homog, np.transpose(pt))
                    dest_point = dest_point/dest_point[2]
                    detected_ball_dst_pos = np.transpose(dest_point)

                    # track ball history
                    if show_b:
                        if len(ball_track_history['src'])>0 :
                            if np.linalg.norm(detected_ball_src_pos-ball_track_history['src'][-1])<ball_track_dist_thresh:
                                ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
                                ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))
                            else:
                                ball_track_history['src']=[(int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1]))]
                                ball_track_history['dst']=[(int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1]))]
                        else:
                            ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
                            ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))
                    
                if len(ball_track_history) > max_track_length:
                    ball_track_history['src'].pop(0)
                    ball_track_history['dst'].pop(0)

                
            ######### Part 2 ########## 
            # Players Team Prediction #
            ###########################

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                      # Convert frame to RGB
            obj_palette_list = []                                                                   # Initialize players color palette list
            palette_interval = (0,num_pal_colors)                                                   # Color interval to extract from dominant colors palette (1rd to 5th color)

            ## Loop over detected players (label 0) and extract dominant colors palette based on defined interval
            for i, j in enumerate(list(results_players[0].boxes.cls.cpu().numpy())):
                if int(j) == 0:
                    bbox = results_players[0].boxes.xyxy.cpu().numpy()[i,:]                         # Get bbox info (x,y,x,y)
                    obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]       # Crop bbox out of the frame
                    obj_img_w, obj_img_h = obj_img.shape[1], obj_img.shape[0]
                    center_filter_x1 = np.max([(obj_img_w//2)-(obj_img_w//5), 1])
                    center_filter_x2 = (obj_img_w//2)+(obj_img_w//5)
                    center_filter_y1 = np.max([(obj_img_h//3)-(obj_img_h//5), 1])
                    center_filter_y2 = (obj_img_h//3)+(obj_img_h//5)
                    center_filter = obj_img[center_filter_y1:center_filter_y2, 
                                            center_filter_x1:center_filter_x2]
                    obj_pil_img = Image.fromarray(np.uint8(center_filter))                          # Convert to pillow image
                    reduced = obj_pil_img.convert("P", palette=Image.Palette.WEB)                   # Convert to web palette (216 colors)
                    palette = reduced.getpalette()                                                  # Get palette as [r,g,b,r,g,b,...]
                    palette = [palette[3*n:3*n+3] for n in range(256)]                              # Group 3 by 3 = [[r,g,b],[r,g,b],...]
                    color_count = [(n, palette[m]) for n,m in reduced.getcolors()]                  # Create list of palette colors with their frequency
                    RGB_df = pd.DataFrame(color_count, columns = ['cnt', 'RGB']).sort_values(       # Create dataframe based on defined palette interval
                                        by = 'cnt', ascending = False).iloc[
                                            palette_interval[0]:palette_interval[1],:]
                    palette = list(RGB_df.RGB)                                                      # Convert palette to list (for faster processing)
                    
                    # Update detected players color palette list
                    obj_palette_list.append(palette)
            
            ## Calculate distances between each color from every detected player color palette and the predefined teams colors
            players_distance_features = []
            # Loop over detected players extracted color palettes
            for palette in obj_palette_list:
                palette_distance = []
                palette_lab = [skimage.color.rgb2lab([i/255 for i in color]) for color in palette]  # Convert colors to L*a*b* space
                # Loop over colors in palette
                for color in palette_lab:
                    distance_list = []
                    # Loop over predefined list of teams colors
                    for c in color_list_lab:
                        #distance = np.linalg.norm([i/255 - j/255 for i,j in zip(color,c)])
                        distance = skimage.color.deltaE_cie76(color, c)                             # Calculate Euclidean distance in Lab color space
                        distance_list.append(distance)                                              # Update distance list for current color
                    palette_distance.append(distance_list)                                          # Update distance list for current palette
                players_distance_features.append(palette_distance)                                  # Update distance features list

            ## Predict detected players teams based on distance features
            players_teams_list = []
            # Loop over players distance features
            for distance_feats in players_distance_features:
                vote_list=[]
                # Loop over distances for each color 
                for dist_list in distance_feats:
                    team_idx = dist_list.index(min(dist_list))//nbr_team_colors                     # Assign team index for current color based on min distance
                    vote_list.append(team_idx)                                                      # Update vote voting list with current color team prediction
                players_teams_list.append(max(vote_list, key=vote_list.count))                      # Predict current player team by vote counting


            #################### Part 3 #####################
            # Updated Frame & Tactical Map With Annotations #
            #################################################
            
            ball_color_bgr = (0,0,255)                                                                          # Color (GBR) for ball annotation on tactical map
            j=0                                                                                                 # Initializing counter of detected players
            palette_box_size = 10                                                                               # Set color box size in pixels (for display)
            annotated_frame = frame                                                                             # Create annotated frame

            # Loop over all detected object by players detection model
            for i in range(bboxes_p.shape[0]):
                conf = confs_p[i]                                                                               # Get confidence of current detected object
                if labels_p[i]==0:                                                                              # Display annotation for detected players (label 0)
                    
                    # Display extracted color palette for each detected player
                    if show_pal:
                        palette = obj_palette_list[j]                                                           # Get color palette of the detected player
                        for k, c in enumerate(palette):
                            c_bgr = c[::-1]                                                                     # Convert color to BGR
                            annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_p[i,2])+3,             # Add color palette annotation on frame
                                                                    int(bboxes_p[i,1])+k*palette_box_size),
                                                                    (int(bboxes_p[i,2])+palette_box_size,
                                                                    int(bboxes_p[i,1])+(palette_box_size)*(k+1)),
                                                                    c_bgr, -1)
                    
                    team_name = list(colors_dic.keys())[players_teams_list[j]]                                  # Get detected player team prediction
                    color_rgb = colors_dic[team_name][0]                                                        # Get detected player team color
                    color_bgr = color_rgb[::-1]                                                                 # Convert color to bgr
                    if show_p:
                        annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_p[i,0]), int(bboxes_p[i,1])),  # Add bbox annotations with team colors
                                                        (int(bboxes_p[i,2]), int(bboxes_p[i,3])), color_bgr, 1)
                        
                        annotated_frame = cv2.putText(annotated_frame, team_name + f" {conf:.2f}",                  # Add team name annotations
                                    (int(bboxes_p[i,0]), int(bboxes_p[i,1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    color_bgr, 2)
                    
                    # Add tactical map player postion color coded annotation if more than 3 field keypoints are detected
                    if 'homog' in locals():
                        tac_map_copy = cv2.circle(tac_map_copy, (int(pred_dst_pts[j][0]),int(pred_dst_pts[j][1])),
                                            radius=5, color=color_bgr, thickness=-1)
                        tac_map_copy = cv2.circle(tac_map_copy, (int(pred_dst_pts[j][0]),int(pred_dst_pts[j][1])),
                                            radius=5, color=(0,0,0), thickness=1)

                    j+=1                                                                                        # Update players counter
                else:                                                                                           # Display annotation for otehr detections (label 1, 2)
                    annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_p[i,0]), int(bboxes_p[i,1])),  # Add white colored bbox annotations
                                                    (int(bboxes_p[i,2]), int(bboxes_p[i,3])), (255,255,255), 1)
                    annotated_frame = cv2.putText(annotated_frame, labels_dic[labels_p[i]] + f" {conf:.2f}",    # Add white colored label text annotations
                                (int(bboxes_p[i,0]), int(bboxes_p[i,1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255,255,255), 2)

                    # Add tactical map ball postion annotation if detected
                    if detected_ball_src_pos is not None and 'homog' in locals():
                        tac_map_copy = cv2.circle(tac_map_copy, (int(detected_ball_dst_pos[0]), 
                                                    int(detected_ball_dst_pos[1])), radius=5, 
                                                    color=ball_color_bgr, thickness=3)
            if show_k:
                for i in range(bboxes_k.shape[0]):
                    annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_k[i,0]), int(bboxes_k[i,1])),  # Add bbox annotations with team colors
                                                (int(bboxes_k[i,2]), int(bboxes_k[i,3])), (0,0,0), 1)
            # Plot the tracks
            if len(ball_track_history['src'])>0:
                points = np.hstack(ball_track_history['dst']).astype(np.int32).reshape((-1, 1, 2))
                tac_map_copy = cv2.polylines(tac_map_copy, [points], isClosed=False, color=(0, 0, 100), thickness=2)

            
            
            # Combine annotated frame and tactical map in one image with colored border separation
            border_color = [255,255,255]                                                                        # Set border color (BGR)
            annotated_frame=cv2.copyMakeBorder(annotated_frame, 40, 10, 10, 10,                                 # Add borders to annotated frame
                                                cv2.BORDER_CONSTANT, value=border_color)
            tac_map_copy = cv2.copyMakeBorder(tac_map_copy, 70, 50, 10, 10, cv2.BORDER_CONSTANT,                # Add borders to tactical map 
                                            value=border_color)      
            tac_map_copy = cv2.resize(tac_map_copy, (tac_map_copy.shape[1], annotated_frame.shape[0]))          # Resize tactical map
            final_img = cv2.hconcat((annotated_frame, tac_map_copy))                                            # Concatenate both images
            ## Add info annotation
            cv2.putText(final_img, "Tactical Map", (1370,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

            new_frame_time = time.time()                                                                        # Get time after finished processing current frame
            fps = 1/(new_frame_time-prev_frame_time)                                                            # Calculate FPS as 1/(frame proceesing duration)
            prev_frame_time = new_frame_time                                                                    # Save current time to be used in next frame
            cv2.putText(final_img, "FPS: " + str(int(fps)), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            
            # Display the annotated frame
            stframe.image(final_img, channels="BGR")
            #cv2.imshow("YOLOv8 Inference", frame)
            if save_output:
                output.write(cv2.resize(final_img, (width, height)))


    # Remove progress bar and return        
    st_prog_bar.empty()
    return True
        
    


    