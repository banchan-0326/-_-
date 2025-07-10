import numpy as np
import warnings

# 빌딩 내의 사람들이 주기적으로 수행할 미션 (회의, 식사시간 등등)

class Mission:
    def __init__(self, target, start_time, end_time):
        self.target = target # 미션 층
        self.start_time = start_time # 미션 시작 시간
        self.end_time = end_time # 미션 종료 시간
        if start_time == end_time:
            raise warnings.warn("[Warning] Start and End time of Mission is same. It may cause unintended result!")

    def is_overlap_with(self, other_mission): # 어떤 미션이 이 미션과 시간이 겹치는지 검사
        # Overlap happens if one mission starts before the other one ends, AND
        # the other mission starts before the first one ends.
        # self: [s1, e1), other_mission: [s2, e2)
        # Overlap if s1 < e2 and s2 < e1
        s1 = self.start_time
        e1 = self.end_time
        s2 = other_mission.start_time
        e2 = other_mission.end_time

        return s1 < e2 and s2 < e1

    def is_valid(self, current_time): # 현재 미션이 유효한 미션인지 검사
        if self.start_time <= current_time < self.end_time:
            return True
        else:
            return False
