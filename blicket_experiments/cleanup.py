import os
import sys

def delete_judge_guess_files(root_path: str):
    if not os.path.isdir(root_path):
        print(f"Error: {root_path} is not a valid directory.")
        return

    deleted = 0

    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith("judge_guess.jsonl"):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    deleted += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

    print(f"\nTotal files deleted: {deleted}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python delete_judge_guess.py <root_directory>")
        sys.exit(1)

    root_dir = sys.argv[1]
    delete_judge_guess_files(root_dir)
