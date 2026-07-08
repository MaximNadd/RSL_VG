import decord
import numpy as np
import cv2

from .util import draw_pose
from .dwpose_detector import dwpose_detector as dwprocessor


def get_video_pose(
        video_path: str,
        ref_image: np.ndarray,
        sample_stride: int=1):
    """preprocess ref image pose and video pose

    Args:
        video_path (str): video pose path
        ref_image (np.ndarray): reference image 
        sample_stride (int, optional): Defaults to 1.

    Returns:
        np.ndarray: sequence of video pose
    """
    # select ref-keypoint from reference pose for pose rescale
    ref_pose = dwprocessor(ref_image)
    ref_keypoint_id = [0, 1, 2, 5, 8, 11, 14, 15, 16, 17]
    ref_keypoint_id = [i for i in ref_keypoint_id \
        if ref_pose['bodies']['score'].shape[0] > 0 and ref_pose['bodies']['score'][0][i] > 0.3]
    ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]

    height, width, _ = ref_image.shape

    # read input video
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    sample_stride *= max(1, int(vr.get_avg_fps() / 24))

    detected_poses = [dwprocessor(frm) for frm in vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()]

    detected_bodies = np.stack(
        [p['bodies']['candidate'] for p in detected_poses if p['bodies']['candidate'].shape[0] == 18])[:,
                      ref_keypoint_id]
    # compute linear-rescale params
    ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
    fh, fw, _ = vr[0].shape
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])
    output_pose = []
    # pose rescale 
    for detected_pose in detected_poses:
        detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
        detected_pose['faces'] = detected_pose['faces'] * a + b
        detected_pose['hands'] = detected_pose['hands'] * a + b
        im = draw_pose(detected_pose, height, width)
        output_pose.append(np.array(im))
    return np.stack(output_pose)


def get_image_pose(ref_image):
    """process image pose

    Args:
        ref_image (np.ndarray): reference image pixel value

    Returns:
        np.ndarray: pose visual image in RGB-mode
    """
    height, width, _ = ref_image.shape
    ref_pose = dwprocessor(ref_image)
    pose_img = draw_pose(ref_pose, height, width)
    return np.array(pose_img)

def get_video_pose_new(
        video_path: str, 
        ref_image: np.ndarray,
        sample_stride: int=1):
    """preprocess ref image pose and video pose

    Args:
        video_path (str): video pose path
        ref_image (np.ndarray): reference image 
        sample_stride (int, optional): Defaults to 1.

    Returns:
        np.ndarray: sequence of video pose
    """


    # read input video
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    sample_stride *= max(1, int(vr.get_avg_fps() / 24))

    # Read frames with the specified stride
    frame_indices = list(range(0, len(vr), sample_stride))
    frames = vr.get_batch(frame_indices).asnumpy()

    # Convert frames from BGR to RGB if needed
    target_height, target_width = ref_image.shape[:2]
    frames_rgb_resized = []
    for frame in frames:
        original_height, original_width = frame.shape[:2]

        # Resize while maintaining aspect ratio
        aspect_ratio = original_width / original_height
        if (target_width / target_height) > aspect_ratio:
            new_height = target_height
            new_width = int(aspect_ratio * target_height)
        else:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)

        resized_frame = cv2.resize(frame, (new_width, new_height))

        # If the resized frame doesn't match the target dimensions exactly, pad with black
        if new_width != target_width or new_height != target_height:
            padded_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            start_x = (target_width - new_width) // 2
            start_y = (target_height - new_height) // 2
            padded_frame[start_y:start_y + new_height, start_x:start_x + new_width] = resized_frame
            resized_frame = padded_frame

        # Create a mask where the white pixels are
        white_mask = cv2.inRange(resized_frame, (240, 240, 240), (255, 255, 255))

        # Set white pixels to black
        resized_frame[white_mask == 255] = (0, 0, 0)

        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        frames_rgb_resized.append(frame_rgb)

    return np.array(frames_rgb_resized)


