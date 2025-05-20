"""
任務執行模組初始化
"""

# 導入所有任務執行函數，使其可以直接從tasks包中引用
from tasks.gender_task import run_gender_task
from tasks.hold_racket_task import run_hold_racket_task
from tasks.play_years_task import run_play_years_task
from tasks.level_task import run_level_task

# 維護任務名稱到執行函數的映射
TASK_RUNNERS = {
    'gender': run_gender_task,
    'hold_racket_handed': run_hold_racket_task,
    'play_years': run_play_years_task,
    'level': run_level_task
}

def run_task(task_name, **kwargs):
    """
    根據任務名稱執行對應的任務
    
    Args:
        task_name (str): 任務名稱
        **kwargs: 傳遞給任務執行函數的參數
        
    Returns:
        執行結果
    """
    if task_name not in TASK_RUNNERS:
        raise ValueError(f"未知的任務名稱: {task_name}")
    
    return TASK_RUNNERS[task_name](**kwargs)