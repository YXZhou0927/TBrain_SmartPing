"""
主程式 - 執行所有分類任務或指定的單個任務
"""

import argparse
import time
import os
from tasks import run_task, TASK_RUNNERS
from config import base_config

def main():
    """主函數"""
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='執行運動員特徵分類任務')
    parser.add_argument('--tasks', nargs='+', choices=['gender', 'hold_racket_handed', 'play_years', 'level', 'all'],
                        default=['all'], help='指定要執行的任務')
    parser.add_argument('--no-gpu', action='store_true', help='禁用GPU加速')
    parser.add_argument('--trials', type=int, help='指定Optuna試驗次數')
    parser.add_argument('--merge', action='store_true', help='執行後自動合併提交文件')
    
    args = parser.parse_args()
    
    # 確定要執行的任務
    tasks = args.tasks
    if 'all' in tasks:
        tasks = list(TASK_RUNNERS.keys())
    
    # 是否使用GPU
    use_gpu = not args.no_gpu
    
    # 記錄總執行時間
    total_start_time = time.time()
    
    # 存儲每個任務的結果
    task_results = {}
    
    # 執行所有指定的任務
    for task in tasks:
        try:
            submission, test_meta = run_task(task, n_trials=args.trials, use_gpu=use_gpu)
            task_results[task] = (submission, test_meta)
        except Exception as e:
            print(f"執行 {task} 任務時出錯: {str(e)}")
    
    # 計算總執行時間
    total_time = time.time() - total_start_time
    print(f"\n所有指定任務執行完成！總耗時: {total_time:.2f} 秒")
    
    # 自動合併提交文件
    if args.merge or len(tasks) > 1:
        try:
            from merge_submissions import merge_all_submissions
            merged_submission = merge_all_submissions()
            print(f"所有任務的提交文件已合併")
        except Exception as e:
            print(f"合併提交文件時出錯: {str(e)}")
    
    return task_results

if __name__ == "__main__":
    main()