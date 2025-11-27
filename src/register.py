import argparse
import os
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='model.pt')
    parser.add_argument('--metric', type=float, default=0.0)
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--out-folder', default='models')
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)
    if args.metric >= args.threshold and os.path.exists(args.model_path):
        dst = os.path.join(args.out_folder, f"model_{args.metric:.4f}.pt")
        shutil.copy(args.model_path, dst)
        print(f"Model registered into {dst}")
    else:
        print(f"Model not registered. metric={args.metric}, threshold={args.threshold}")
