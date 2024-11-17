# Import necessary libraries
from utils import (read_video, save_video, measure_distance, draw_player_stat, convert_pixel_distance_to_meters)
import constants
from trackers import (PlayerTracker, BallTracker)
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
from copy import deepcopy
import pandas as pd

def main():
    # Read Video
    input_video_path = "C:\\Users\\91868\\Desktop\\input\\input_video.mp4"
    video_frames = read_video(input_video_path)
    print(f"Number of frames read: {len(video_frames)}")

    # Trackers and models initialization
    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker = BallTracker(model_path="C:\\Users\\91868\\tennis_analysis\\models\\yolo5_last.pt")
    court_line_detector = CourtLineDetector("C:\\Users\\91868\\tennis_analysis\\models\\keypoints_model.pth")

    # Detect players and ball
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, 
                                                     stub_path="C:\\Users\\91868\\tennis_analysis\\tracker_stubs\\player_detection.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, 
                                                 stub_path="C:\\Users\\91868\\tennis_analysis\\tracker_stubs\\ball_detection.pkl")
    ball_detections = ball_tracker.interpolate_ball_position(ball_detections)

    # Detect court keypoints
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Choose players based on court keypoints
    player_detections = player_tracker.choose_and_filter_player(court_keypoints, player_detections)

    # Initialize MiniCourt for analysis
    mini_court = MiniCourt(video_frames[0])

    # Detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frame(ball_detections)
    print(ball_shot_frames)

    # Convert positions to mini court coordinates
    player_mini_court_detection, ball_mini_court_detection = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, ball_detections, court_keypoints)

    # Initialize player statistics data
    player_stat_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,
        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0
    }]

    # Analyze ball shots for player statistics
    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # 24 frames per second

        # Calculate distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detection[start_frame][1],
                                                           ball_mini_court_detection[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court())

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        # Identify player who shot the ball
        player_positions = player_mini_court_detection[start_frame]
        player_shot_ball = min(player_positions.keys(), 
                                key=lambda player_id: measure_distance(player_positions[player_id],
                                                                        ball_mini_court_detection[start_frame][1]))

        # Opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detection[start_frame][opponent_player_id],
                                                               player_mini_court_detection[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(distance_covered_by_opponent_pixels,
                                                                               constants.DOUBLE_LINE_WIDTH,
                                                                               mini_court.get_width_of_mini_court())

        speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6

        current_player_stats = deepcopy(player_stat_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stat_data.append(current_player_stats)

    # Create DataFrame for player statistics
    player_stat_data_df = pd.DataFrame(player_stat_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stat_data_df = pd.merge(frames_df, player_stat_data_df, on='frame_num', how='left')
    player_stat_data_df = player_stat_data_df.ffill()

    # Calculate average speeds
    player_stat_data_df['player_1_average_shot_speed'] = player_stat_data_df['player_1_total_shot_speed'] / player_stat_data_df['player_1_number_of_shots']
    player_stat_data_df['player_2_average_shot_speed'] = player_stat_data_df['player_2_total_shot_speed'] / player_stat_data_df['player_2_number_of_shots']
    player_stat_data_df['player_1_average_player_speed'] = player_stat_data_df['player_1_total_player_speed'] / player_stat_data_df['player_1_number_of_shots']
    player_stat_data_df['player_2_average_player_speed'] = player_stat_data_df['player_2_total_player_speed'] / player_stat_data_df['player_2_number_of_shots']

    # Draw Bounding Boxes and Statistics on Frames
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detection)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detection, color=(0, 255, 255))
    output_video_frames = draw_player_stat(output_video_frames, player_stat_data_df)

    # Draw frame number on each frame
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Save final output video
    save_video(output_video_frames, "C:\\Users\\91868\\Desktop\\tennis_ball_project\\newoutput\\output_video_3.avi")

if __name__ == "__main__":
    main()
