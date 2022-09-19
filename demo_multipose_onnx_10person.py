#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import time
import argparse
from math import (
    pi,
    atan2,
    sin,
    cos,
    floor,
    degrees,
)

import cv2 as cv
import numpy as np
import onnxruntime


from utils import (
    pad_image,
    rotate_and_crop_rectangle,
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=480)
    parser.add_argument('--mirror', action='store_true')
    parser.add_argument("--keypoint_score", type=float, default=0.20)
    parser.add_argument("--bbox_score", type=float, default=0.20)
    parser.add_argument("--palm_square_crop", action='store_true')
    parser.add_argument("--palm_detection_input_size", choices=[128, 192], type=int, default=192)
    parser.add_argument("--palm_detection_score", type=float, default=0.60)
    args = parser.parse_args()
    return args


def normalize_radians(
    angle: float
) -> float:
    """__normalize_radians
    Parameters
    ----------
    angle: float
    Returns
    -------
    normalized_angle: float
    """
    return angle - 2 * pi * floor((angle + pi) / (2 * pi))


def run_inference_movenet(
    onnx_session,
    input_height,
    input_width,
    input_name0,
    input_name1,
    input_name2,
    image,
):
    image_width = np.asarray(image.shape[1], dtype=np.int64)
    image_height = np.asarray(image.shape[0], dtype=np.int64)
    input_image = copy.deepcopy(image)
    input_image = cv.resize(input_image, dsize=(input_width, input_height))
    input_image = input_image[..., ::-1]
    input_image = input_image.transpose(2,0,1)[np.newaxis, ...].astype(np.float32)
    keypoints_with_scores = onnx_session.run(
        None,
        {
            input_name0: input_image,
            input_name1: image_height,
            input_name2: image_width,
        }
    )[0]
    keypoints_with_scores = np.squeeze(keypoints_with_scores)
    return keypoints_with_scores

def run_inference_palm_detection(
    onnx_session,
    input_height,
    input_width,
    input_name0,
    image,
    keypoints_with_scores,
    bbox_score_th,
    palm_detection_score_th,
    mirror,
):
    image_width = np.asarray(image.shape[1], dtype=np.int64)
    image_height = np.asarray(image.shape[0], dtype=np.int64)

    batch_nums = np.asarray([])
    score_cx_cy_w_wristcenterxy_middlefingerxys = np.asarray([])
    hand_images = []
    for idx, keypoints_with_score in enumerate(keypoints_with_scores):
        if keypoints_with_score[55] > bbox_score_th:
            bbox_x1 = int(keypoints_with_score[51])
            bbox_y1 = int(keypoints_with_score[52])
            bbox_x2 = int(keypoints_with_score[53])
            bbox_y2 = int(keypoints_with_score[54])
            bbox_h = bbox_y2 - bbox_y1
            """
            0:nose,
            1:left eye,
            2:right eye,
            3:left ear,
            4:right ear,
            5:left shoulder,
            6:right shoulder,
            7:left elbow,
            8:right elbow,
            9:left wrist,
            10:right wrist,
            11:left hip,
            12:right hip,
            13:left knee,
            14:right knee,
            15:left ankle,
            16:right ankle
            """
            # 入力画像が判定しているときは左手系と右手系を入れ替える
            rev = 1 if mirror else 0
            elbow_left_x = int(keypoints_with_score[21+rev*3]) # 左肘のX座標
            elbow_left_y = int(keypoints_with_score[22+rev*3]) # 左肘のY座標
            elbow_right_x = int(keypoints_with_score[24-rev*3]) # 右肘のX座標
            elbow_right_y = int(keypoints_with_score[25-rev*3]) # 右肘のY座標
            wrist_left_x = int(keypoints_with_score[27+rev*3]) # 左手首のX座標
            wrist_left_y = int(keypoints_with_score[28+rev*3]) # 左手首のY座標
            wrist_right_x = int(keypoints_with_score[30-rev*3]) # 右手首のX座標
            wrist_right_y = int(keypoints_with_score[31-rev*3]) # 右手首のY座標

            """
            ・左肘と左手首のX座標の位置関係を見て横方向のクロップ位置を微妙に補正する
                左肘X座標 > 左手首Y座標: クロップ領域を画角左方向に少しずらし補正
                左肘X座標 = 左手首X座標: ずらし補正なし
                左肘X座標 < 左手首X座標: クロップ領域を画角右方向に少しずらし補正

            ・左肘と左手首のY座標の位置関係を見て縦方向のクロップ位置を微妙に補正する
                左肘Y座標 > 左手首Y座標: クロップ領域を画角上方向に少しずらし補正
                左肘Y座標 = 左手首Y座標: ずらし補正なし
                左肘Y座標 < 左手首Y座標: クロップ領域を画角上方向に少しずらし補正
            """
            distx_left_elbow_to_left_wrist = elbow_left_x - wrist_left_x # +:肘>手首, -:肘<手首
            disty_left_elbow_to_left_wrist = elbow_left_y - wrist_left_y # +:肘が下で手首が上, -:肘が上で手首が下
            distx_right_elbow_to_right_wrist = elbow_right_x - wrist_right_x # +:肘>手首, -:肘<手首
            disty_right_elbow_to_right_wrist = elbow_right_y - wrist_right_y # +:肘が下で手首が上, -:肘が上で手首が下
            adjust_ratio = 2

            ############################################################## 左手
            # 左肘と左手首のX座標位置関係
            left_wrist_x_adjust_pixel = 0
            inversion = -1 if mirror else 1
            if distx_left_elbow_to_left_wrist > 0:
                left_wrist_x_adjust_pixel = (distx_left_elbow_to_left_wrist // adjust_ratio) * inversion
            elif distx_left_elbow_to_left_wrist == 0:
                left_wrist_x_adjust_pixel = 0
            elif  distx_left_elbow_to_left_wrist < 0:
                left_wrist_x_adjust_pixel = (distx_left_elbow_to_left_wrist // adjust_ratio) * inversion
            # 左肘と左手首のY座標位置関係
            left_wrist_y_adjust_pixel = 0
            if disty_left_elbow_to_left_wrist > 0:
                left_wrist_y_adjust_pixel = (disty_left_elbow_to_left_wrist // adjust_ratio) * -1
            elif disty_left_elbow_to_left_wrist == 0:
                left_wrist_y_adjust_pixel = 0
            elif  disty_left_elbow_to_left_wrist < 0:
                left_wrist_y_adjust_pixel = (disty_left_elbow_to_left_wrist // adjust_ratio) * -1
            # クロップ中心位置補正
            wrist_left_x = wrist_left_x + left_wrist_x_adjust_pixel
            wrist_left_y = wrist_left_y + left_wrist_y_adjust_pixel
            # 正方形のクロップ領域を crop_magnification倍 に拡張する
            crop_magnification = 1.0
            wrist_left_x1 = wrist_left_x - (bbox_h / 4 * crop_magnification) # 左手手首の中心座標から肩幅の半分左にずらした点
            wrist_left_y1 = wrist_left_y - (bbox_h / 4 * crop_magnification) # 左手手首の中心座標から肩幅の半分上にずらした点
            wrist_left_x2 = wrist_left_x + (bbox_h / 4 * crop_magnification) # 左手手首の中心座標から肩幅の半分右にずらした点
            wrist_left_y2 = wrist_left_y + (bbox_h / 4 * crop_magnification) # 左手手首の中心座標から肩幅の半分下にずらした点
            # 画角の範囲外参照回避
            wrist_left_x1 = int(min(max(0, wrist_left_x1), image_width))
            wrist_left_y1 = int(min(max(0, wrist_left_y1), image_height))
            wrist_left_x2 = int(min(max(0, wrist_left_x2), image_width))
            wrist_left_y2 = int(min(max(0, wrist_left_y2), image_height))
            # 四方をパディングして正方形にした画像の取得
            square_crop_size = max(wrist_left_x2 - wrist_left_x1, wrist_left_y2 - wrist_left_y1)
            left_padded_image = pad_image(
                image=image[wrist_left_y1:wrist_left_y2, wrist_left_x1:wrist_left_x2, :],
                resize_width=square_crop_size,
                resize_height=square_crop_size,
            )
            if left_padded_image.shape[0] > 0 and left_padded_image.shape[1] > 0:
                left_padded_image_resized = cv.resize(
                    left_padded_image,
                    dsize=(input_width, input_height),
                )
                hand_images.append(left_padded_image_resized)

            ############################################################## 右手
            # 左肘と左手首のX座標位置関係
            right_wrist_x_adjust_pixel = 0
            inversion = -1 if mirror else 1
            if distx_right_elbow_to_right_wrist > 0:
                right_wrist_x_adjust_pixel = (distx_right_elbow_to_right_wrist // adjust_ratio) * inversion
            elif distx_right_elbow_to_right_wrist == 0:
                right_wrist_x_adjust_pixel = 0
            elif  distx_right_elbow_to_right_wrist < 0:
                right_wrist_x_adjust_pixel = (distx_right_elbow_to_right_wrist // adjust_ratio) * inversion
            # 左肘と左手首のY座標位置関係
            right_wrist_y_adjust_pixel = 0
            if disty_right_elbow_to_right_wrist > 0:
                right_wrist_y_adjust_pixel = (disty_right_elbow_to_right_wrist // adjust_ratio) * -1
            elif disty_right_elbow_to_right_wrist == 0:
                right_wrist_y_adjust_pixel = 0
            elif  disty_right_elbow_to_right_wrist < 0:
                right_wrist_y_adjust_pixel = (disty_right_elbow_to_right_wrist // adjust_ratio) * -1
            # クロップ中心位置補正
            wrist_right_x = wrist_right_x + right_wrist_x_adjust_pixel
            wrist_right_y = wrist_right_y + right_wrist_y_adjust_pixel
            # 正方形のクロップ領域を crop_magnification倍 に拡張する
            crop_magnification = 1.0
            wrist_right_x1 = wrist_right_x - (bbox_h / 4 * crop_magnification) # 左手手首の中心座標から肩幅の半分左にずらした点
            wrist_right_y1 = wrist_right_y - (bbox_h / 4 * crop_magnification) # 左手手首の中心座標から肩幅の半分上にずらした点
            wrist_right_x2 = wrist_right_x + (bbox_h / 4 * crop_magnification) # 左手手首の中心座標から肩幅の半分右にずらした点
            wrist_right_y2 = wrist_right_y + (bbox_h / 4 * crop_magnification) # 左手手首の中心座標から肩幅の半分下にずらした点
            # 画角の範囲外参照回避
            wrist_right_x1 = int(min(max(0, wrist_right_x1), image_width))
            wrist_right_y1 = int(min(max(0, wrist_right_y1), image_height))
            wrist_right_x2 = int(min(max(0, wrist_right_x2), image_width))
            wrist_right_y2 = int(min(max(0, wrist_right_y2), image_height))
            # 四方をパディングして正方形にした画像の取得
            square_crop_size = max(wrist_right_x2 - wrist_right_x1, wrist_right_y2 - wrist_right_y1)
            right_padded_image = pad_image(
                image=image[wrist_right_y1:wrist_right_y2, wrist_right_x1:wrist_right_x2, :],
                resize_width=square_crop_size,
                resize_height=square_crop_size,
            )
            if right_padded_image.shape[0] > 0 and right_padded_image.shape[1] > 0:
                right_padded_image_resized = cv.resize(
                    right_padded_image,
                    dsize=(input_width, input_height),
                )
                hand_images.append(right_padded_image_resized)


    hand_info_list = []
    hand_image_list = []
    if len(hand_images) > 0:
        input_images = np.asarray(hand_images, dtype=np.float32)
        input_images = input_images / 255.0
        input_images = input_images[..., ::-1]
        input_images = input_images.transpose(0,3,1,2)
        batch_nums, score_cx_cy_w_wristcenterxy_middlefingerxys = onnx_session.run(
            None,
            {
                input_name0: input_images,
            }
        )
        keep = score_cx_cy_w_wristcenterxy_middlefingerxys[:, 0] > palm_detection_score_th
        score_cx_cy_w_wristcenterxy_middlefingerxys = score_cx_cy_w_wristcenterxy_middlefingerxys[keep, :]
        batch_nums = batch_nums[keep, :]
        input_images = [input_images[idx] for idx in batch_nums]

        for input_image, score_cx_cy_w_wristcenterxy_middlefingerxy in zip(input_images, score_cx_cy_w_wristcenterxy_middlefingerxys):
            cx = score_cx_cy_w_wristcenterxy_middlefingerxy[1]
            cy = score_cx_cy_w_wristcenterxy_middlefingerxy[2]
            w = score_cx_cy_w_wristcenterxy_middlefingerxy[3]
            wrist_center_x = score_cx_cy_w_wristcenterxy_middlefingerxy[4]
            wrist_center_y = score_cx_cy_w_wristcenterxy_middlefingerxy[5]
            middlefinger_x = score_cx_cy_w_wristcenterxy_middlefingerxy[6]
            middlefinger_y = score_cx_cy_w_wristcenterxy_middlefingerxy[7]

            if w > 0:
                kp02_x = middlefinger_x - wrist_center_x
                kp02_y = middlefinger_y - wrist_center_y
                extended_area_size = 2.9 * w
                rotation = 0.5 * pi - atan2(-kp02_y, kp02_x) # radians
                rotation = normalize_radians(rotation)
                cx = cx + 0.5*w*sin(rotation)
                cy = cy - 0.5*w*cos(rotation)
                degree = degrees(rotation) # radians to degrees

                # sqn_rr_center_y = (sqn_rr_center_y * self.square_standard_size - self.square_padding_half_size) / image_height
                # hands.append(
                #     [
                #         sqn_rr_size,
                #         rotation,
                #         sqn_rr_center_x,
                #         sqn_rr_center_y,
                #     ]
                # )
                # print(f'score:{score} cx:{cx} cy:{cy} sqn_rr_size:{sqn_rr_size} rotation:{rotation} sqn_rr_center_x:{sqn_rr_center_x} sqn_rr_center_y:{sqn_rr_center_y}')

                hand_image = input_image * 255 # 0-255
                hand_image = hand_image[0].transpose(1,2,0)[..., ::-1].astype(np.uint8) # RGB
                hand_image_height = hand_image.shape[0]
                hand_image_width = hand_image.shape[1]

                debug_image = copy.deepcopy(hand_image)

                ################# debug
                # Palm Detection で検出した手のひらの中心点の描画
                cv.circle(
                    debug_image,
                    (int(cx*hand_image_width), int(cy*hand_image_height)),
                    3,
                    (0, 0, 255),
                    -1
                )
                ################# debug

                rcx = cx * hand_image_width
                rcy = cy * hand_image_height
                x1 = int((cx - 0.5 * extended_area_size) * hand_image_width)
                y1 = int((cy - 0.5 * extended_area_size) * hand_image_height)
                x2 = int((cx + 0.5 * extended_area_size) * hand_image_width)
                y2 = int((cy + 0.5 * extended_area_size) * hand_image_height)
                x1 = min(max(0, x1), hand_image_width)
                y1 = min(max(0, y1), hand_image_height)
                x2 = min(max(0, x2), hand_image_width)
                y2 = min(max(0, y2), hand_image_height)

                ################# debug
                # Palm Detection で検出した手のひらのバウンディングボックス描画(回転非考慮), オレンジ色の枠
                cv.rectangle(
                    debug_image,
                    (x1,y1),
                    (x2,y2),
                    (0,128,255),
                    2,
                    cv.LINE_AA,
                )
                ################# debug

                # [元画像の中心座標X, 元画像の中心座標Y, 元画像スケールの手のひらの幅, 元画像スケールの手のひらの高さ, 回転角度]
                hand_info = [rcx, rcy, (x2-x1), (y2-y1), degree]
                hand_info_list.append(hand_info)
                hand_info_np = np.asarray(hand_info, dtype=np.float32)

                ################# debug
                # 回転を考慮したバウンディングボックスの描画, 赤色の枠
                hand_info_tuple = ((hand_info_np[0], hand_info_np[1]), (hand_info_np[2], hand_info_np[3]), hand_info_np[4])
                box = cv.boxPoints(hand_info_tuple).astype(np.int0)
                cv.drawContours(debug_image, [box], 0,(0,0,255), 2, cv.LINE_AA)
                ################# debug

                # クロップ済み、かつ、回転角ゼロ度に調整された画像のリスト(Hand Landmarkモデル入力用画像)
                # 常時１件のリスト
                cropted_rotated_hands_images = rotate_and_crop_rectangle(
                    image=hand_image,
                    hand_info_nps=hand_info_np[np.newaxis, ...],
                    operation_when_cropping_out_of_range='padding',
                )
                hand_image_list.append(cropted_rotated_hands_images[0])

                ################# debug
                cv.imshow(f'no_rotated', debug_image)
                cv.imshow(f'rotated', cropted_rotated_hands_images[0])
                ################# debug

        # hand_info_list をListからnp.ndarrayに変換 [N, 5]
        # [元画像の中心座標X, 元画像の中心座標Y, 元画像スケールの手のひらの幅, 元画像スケールの手のひらの高さ, 回転角度]
        hand_info_list = np.asarray(hand_info_list, dtype=np.float32)

    return batch_nums, score_cx_cy_w_wristcenterxy_middlefingerxys, hand_info_list, hand_image_list


def run_inference_handlandmark_detection(
    onnx_session,
    input_height,
    input_width,
    input_name0,
    image,
):
    pass


def main():
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device

    if args.file is not None:
        cap_device = args.file

    cap = cv.VideoCapture(cap_device)

    if args.file is not None:
        cap_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        cap_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    else:
        cap_width = args.width
        cap_height = args.height
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mirror = args.mirror
    keypoint_score_th = args.keypoint_score
    bbox_score_th = args.bbox_score
    palm_square_crop = args.palm_square_crop
    palm_detection_input_size = args.palm_detection_input_size
    palm_detection_score_th = args.palm_detection_score

    # カメラ準備 ###############################################################
    cap_fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc('m','p','4','v')
    video_writer = cv.VideoWriter(
        filename='output.mp4',
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(cap_width, cap_height),
    )

    # モデルロード #############################################################
    # MoveNet
    mn_model_path = f"onnx/movenet_multipose_lightning_256x320_p20.onnx"
    mn_onnx_session = onnxruntime.InferenceSession(
        mn_model_path,
        providers=[
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )
    mn_input_name0 = mn_onnx_session.get_inputs()[0].name
    _, _, mn_input_height, mn_input_width = mn_onnx_session.get_inputs()[0].shape
    mn_input_name1 = mn_onnx_session.get_inputs()[1].name
    mn_input_name2 = mn_onnx_session.get_inputs()[2].name

    # Palm Detection
    full_lite_txt = '_full' if palm_detection_input_size == 192 else ''
    pd_model_path = f"onnx/palm_detection{full_lite_txt}_Nx3x{palm_detection_input_size}x{palm_detection_input_size}_post.onnx"
    pd_onnx_session = onnxruntime.InferenceSession(
        pd_model_path,
        providers=[
            # (
            #     'TensorrtExecutionProvider', {
            #         'trt_engine_cache_enable': True,
            #         'trt_engine_cache_path': '.',
            #         'trt_fp16_enable': True,
            #     }
            # ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )
    pd_input_name0 = pd_onnx_session.get_inputs()[0].name
    _, _, pd_input_height, pd_input_width = pd_onnx_session.get_inputs()[0].shape

    # Hand Landmark Detection
    hl_model_path = f"onnx/hand_landmark_sparse_Nx3x224x224.onnx"
    hl_onnx_session = onnxruntime.InferenceSession(
        hl_model_path,
        providers=[
            # (
            #     'TensorrtExecutionProvider', {
            #         'trt_engine_cache_enable': True,
            #         'trt_engine_cache_path': '.',
            #         'trt_fp16_enable': True,
            #     }
            # ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )
    hl_input_name0 = hl_onnx_session.get_inputs()[0].name
    _, _, hl_input_height, hl_input_width = hl_onnx_session.get_inputs()[0].shape


    while True:
        # カメラキャプチャ #####################################################
        ret, frame = cap.read()
        if not ret:
            break
        if mirror:
            frame = cv.flip(frame, 1)  # ミラー表示
        debug_image = copy.deepcopy(frame)

        # 検出実施 ##############################################################
        start_time = time.time()
        keypoints_with_scores = run_inference_movenet(
            mn_onnx_session,
            mn_input_height,
            mn_input_width,
            mn_input_name0,
            mn_input_name1,
            mn_input_name2,
            frame,
        )
        batch_nums = np.asarray([])
        score_cx_cy_w_wristcenterxy_middlefingerxy = np.asarray([])
        if palm_square_crop:
            batch_nums, score_cx_cy_w_wristcenterxy_middlefingerxys, hand_info_list, hand_image_list = run_inference_palm_detection(
                pd_onnx_session,
                pd_input_height,
                pd_input_width,
                pd_input_name0,
                frame,
                keypoints_with_scores,
                bbox_score_th,
                palm_detection_score_th,
                mirror,
            )
            run_inference_handlandmark_detection(
                hl_onnx_session,
                hl_input_height,
                hl_input_width,
                hl_input_name0,
                frame,
            )


        elapsed_time = time.time() - start_time

        # デバッグ描画
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            keypoint_score_th,
            bbox_score_th,
            keypoints_with_scores,
            mirror,
            palm_detection_input_size,
        )

        # キー処理(ESC：終了) ##################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('MoveNet(multipose) Demo', debug_image)
        video_writer.write(debug_image)

    if video_writer:
        video_writer.release()
    if cap:
        cap.release()
    cv.destroyAllWindows()


lines = [
    [0,1],
    [0,2],
    [1,3],
    [2,4],
    [0,5],
    [0,6],
    [5,6],
    [5,7],
    [7,9],
    [6,8],
    [8,10],
    [11,12],
    [5,11],
    [11,13],
    [13,15],
    [6,12],
    [12,14],
    [14,16],
]

def draw_debug(
    image,
    elapsed_time,
    keypoint_score_th,
    bbox_score_th,
    keypoints_with_scores,
    mirror,
    palm_detection_input_size,
):
    debug_image = copy.deepcopy(image)

    """
    0:鼻 1:左目 2:右目 3:左耳 4:右耳 5:左肩 6:右肩 7:左肘 8:右肘 # 9:左手首
    10:右手首 11:左股関節 12:右股関節 13:左ひざ 14:右ひざ 15:左足首 16:右足首

    [persons, kpxkpykpscore_x17_bx1by1bx2by2bscore] [10,56]

    0:keypoint0_x
    1:keypoint0_y
    2:keypoint0_score
        :
    46:keypoint16_x
    47:keypoint16_y
    50:keypoint16_score

    51:bbox_x1
    52:bbox_y1
    53:bbox_x2
    54:bbox_y2
    55:bbox_score
    """

    hand_images = []
    for idx, keypoints_with_score in enumerate(keypoints_with_scores):
        if keypoints_with_score[55] > bbox_score_th:
            # Line: bone
            _ = [
                cv.line(
                    debug_image,
                    (int(keypoints_with_score[line_idxs[0]*3+0]), int(keypoints_with_score[line_idxs[0]*3+1])),
                    (int(keypoints_with_score[line_idxs[1]*3+0]), int(keypoints_with_score[line_idxs[1]*3+1])),
                    (0, 255, 0),
                    2
                ) for line_idxs in lines \
                    if keypoints_with_score[line_idxs[0]*3+2] > keypoint_score_th and keypoints_with_score[line_idxs[1]*3+2] > keypoint_score_th
            ]

            # Circle：各点
            _ = [
                cv.circle(
                    debug_image,
                    (int(keypoints_with_score[keypoint_idx*3+0]), int(keypoints_with_score[keypoint_idx*3+1])),
                    3,
                    (0, 0, 255),
                    -1
                ) for keypoint_idx in range(17) if keypoints_with_score[keypoint_idx*3+2] > keypoint_score_th
            ]

            bbox_x1 = int(keypoints_with_score[51])
            bbox_y1 = int(keypoints_with_score[52])
            bbox_x2 = int(keypoints_with_score[53])
            bbox_y2 = int(keypoints_with_score[54])
            bbox_h = bbox_y2 - bbox_y1

            # バウンディングボックス
            cv.rectangle(
                debug_image,
                (bbox_x1, bbox_y1),
                (bbox_x2, bbox_y2),
                (255, 255, 255),
                4,
            )
            cv.rectangle(
                debug_image,
                (bbox_x1, bbox_y1),
                (bbox_x2, bbox_y2),
                (0, 0, 0),
                2,
            )

    # 手のひら検出
    for hand_image in hand_images:
        pass

    # 処理時間
    txt = f"Elapsed Time : {elapsed_time * 1000:.1f} ms (inference + post-process)"
    cv.putText(
        debug_image,
        txt,
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        debug_image,
        txt,
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv.LINE_AA,
    )

    return debug_image


if __name__ == '__main__':
    main()
