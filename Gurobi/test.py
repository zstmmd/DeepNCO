from Gurobi.sp4 import SP4_Robot_Router

if __name__ == "__main__":
    import os
    import sys

    # 导入你原有的类
    from entity.point import Point
    from entity.robot import Robot
    from entity.station import Station
    from entity.stack import Stack
    from entity.task import Task
    from entity.subTask import SubTask
    from problemDto.ofs_problem_dto import OFSProblemDTO


    def create_mock_entity(cls, **kwargs):
        """Python黑魔法：绕过 __init__ 直接创建对象并赋值，防止参数不匹配报错"""
        obj = cls.__new__(cls)
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj


    print("\n" + "=" * 60)
    print("🚚 SP4 物理逻辑微型验证 (Rank 放宽逻辑测试)")
    print("=" * 60)

    # 1. 构造地图点位 (y轴直线)
    # 起点 (0,0) ---> Stack 2 (0,2) ---> Stack 1 (0,8) ---> Station (0,10)
    pt_start = create_mock_entity(Point, idx=0, x=0, y=0)
    pt_station = create_mock_entity(Point, idx=1, x=0, y=10)
    pt_stack1 = create_mock_entity(Point, idx=2, x=0, y=8)  # 离终点近
    pt_stack2 = create_mock_entity(Point, idx=3, x=0, y=2)  # 离起点近
    pt_stack3=create_mock_entity(Point, idx=4, x=1, y=3)
    # 2. 构造基础实体
    robot = create_mock_entity(Robot, id=0, start_point=pt_start)
    station = create_mock_entity(Station, id=0, point=pt_station)
    stack1 = create_mock_entity(Stack, stack_id=1, store_point=pt_stack1)
    stack2 = create_mock_entity(Stack, stack_id=2, store_point=pt_stack2)
    stack3=create_mock_entity(Stack, stack_id=3, store_point=pt_stack3)
    problem = create_mock_entity(OFSProblemDTO)
    problem.robot_list = [robot]
    problem.station_list = [station]
    problem.stack_list = [stack1, stack2,stack3]
    problem.point_to_stack = {1: stack1, 2: stack2,3:stack3}

    # 3. 构造物理任务 (Task) 与 子任务 (SubTask)

    # Task A：位于 Stack 1 (0,8)。它很紧急 (Rank = 0)
    taskA = create_mock_entity(Task,
                               task_id=101, sub_task_id=1, target_stack_id=1, target_station_id=0,
                               robot_service_time=2.0, target_tote_ids=[0,1,3,5], sku_pick_count=1,
                               station_sequence_rank=0
                               )
    taskC = create_mock_entity(Task,task_id=103, sub_task_id=1, target_stack_id=3, target_station_id=0,robot_service_time=2.0,
                               target_tote_ids=[0,1,3,5], sku_pick_count=1,
                               station_sequence_rank=1)
    subTaskA = create_mock_entity(SubTask,
                                  id=1, assigned_station_id=0, station_sequence_rank=0, execution_tasks=[taskA,taskC]
                                  )

    # Task B：位于 Stack 2 (0,2)。它不紧急 (Rank = 1)
    taskB = create_mock_entity(Task,
                               task_id=102, sub_task_id=2, target_stack_id=2, target_station_id=0,
                               robot_service_time=2.0, target_tote_ids=[2], sku_pick_count=1,
                               station_sequence_rank=1
                               )
    subTaskB = create_mock_entity(SubTask,
                                  id=2, assigned_station_id=0, station_sequence_rank=1, execution_tasks=[taskB]
                                  )

    sub_tasks = [subTaskA, subTaskB]

    # 4. 执行测试
    print(">>> 场景设定：")
    print("    - 机器人从 (0,0) 出发")
    print("    - Task B 不紧急(Rank 1)，但离得近 (0,2)")
    print("    - Task A 很紧急(Rank 0)，但离得远 (0,8)")
    print("    - 最终都要送到工作站 (0,10)")
    print(">>> 预期行为：机器人会顺路先拿 Task B，再去拿 Task A。但在工作站会先卸货 A，再卸货 B。\n")

    sp4 = SP4_Robot_Router(problem)
    result_times, result_assign = sp4.solve(sub_tasks, use_mip=True)

    print("\n>>> 结论分析：")
    arr_A = result_times.get(pt_stack1.idx, -1)
    arr_B = result_times.get(pt_stack2.idx, -1)

    if arr_B < arr_A:
        print("✅ 验证成功：机器人放宽了取货顺序，聪明地【先】拿了顺路的 Task B！")
    else:
        print("❌ 验证失败：机器人死板地绕远路先去拿了 Task A。")