#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=480)
    parser.add_argument('--mirror', action='store_true')
    parser.add_argument("--keypoint_score", type=float, default=0.2)
    parser.add_argument("--bbox_score", type=float, default=0.2)

    args = parser.parse_args()

    return args


def run_inference(
    onnx_session,
    input_height,
    input_width,
    input_name0,
    input_name1,
    input_name2,
    image,
):
    image_width, image_height = np.asarray(image.shape[1], dtype=np.int64), np.asarray(image.shape[0], dtype=np.int64)
    input_image = cv.resize(image, dsize=(input_width, input_height))
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
    model_path = f"onnx/movenet_multipose_lightning_256x320_p10.onnx"
    onnx_session = onnxruntime.InferenceSession(
        model_path,
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
    input_name0 = onnx_session.get_inputs()[0].name
    _, _, input_height, input_width = onnx_session.get_inputs()[0].shape
    input_name1 = onnx_session.get_inputs()[1].name
    input_name2 = onnx_session.get_inputs()[2].name

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
        keypoints_with_scores = run_inference(
            onnx_session,
            input_height,
            input_width,
            input_name0,
            input_name1,
            input_name2,
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

    for keypoints_with_score in keypoints_with_scores:
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

            # バウンディングボックス
            cv.rectangle(
                debug_image,
                (int(keypoints_with_score[51]), int(keypoints_with_score[52])),
                (int(keypoints_with_score[53]), int(keypoints_with_score[54])),
                (255, 255, 255),
                4,
            )
            cv.rectangle(
                debug_image,
                (int(keypoints_with_score[51]), int(keypoints_with_score[52])),
                (int(keypoints_with_score[53]), int(keypoints_with_score[54])),
                (0, 0, 0),
                2,
            )

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
