"""Create a Y4M fake-camera fixture from a local, consented video."""
import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=29.0)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.video))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or source_fps <= 0 or args.fps <= 0:
        raise ValueError("invalid video or frame rate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start_frame = int(round(args.start * source_fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current = start_frame - 1
    frames = 0
    try:
        with args.output.open("wb") as stream:
            stream.write(f"YUV4MPEG2 W512 H288 F{args.fps}:1 Ip A1:1 C420jpeg\n".encode("ascii"))
            for sample in range(int(args.seconds * args.fps)):
                target = int(round((args.start + sample / args.fps) * source_fps))
                while current < target:
                    if not capture.grab():
                        raise ValueError("video ends before requested fixture duration")
                    current += 1
                ok, frame = capture.retrieve()
                if not ok:
                    raise ValueError("cannot decode camera fixture")
                frame = cv2.resize(frame, (512, 288), interpolation=cv2.INTER_LINEAR)
                stream.write(b"FRAME\n")
                stream.write(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420).tobytes())
                frames += 1
    finally:
        capture.release()
    print(f"{frames} frames; source={source_fps}fps, replay={args.fps}fps; {args.output}")


if __name__ == "__main__":
    main()
