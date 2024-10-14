#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WRS環境内でロボットを動作させるためのメインプログラム
"""

from __future__ import unicode_literals, print_function, division, absolute_import
import json
import os
from select import select
import traceback
from turtle import pos
import rospy
import rospkg
import tf2_ros
from std_msgs.msg import String
from detector_msgs.srv import (
    SetTransformFromBBox, SetTransformFromBBoxRequest,
    GetObjectDetection, GetObjectDetectionRequest)
from wrs_algorithm.util import omni_base, whole_body, gripper
import math


class WrsMainController(object):
    """
    WRSのシミュレーション環境内でタスクを実行するクラス
    """
    IGNORE_LIST = []
    # "small_marker", "large_marker", "lego_duplo", "spatula", "nine_hole_peg_test"
    GRASP_TF_NAME = "object_grasping"
    GRASP_BACK_SAFE = {"z": 0.05, "xy": 0.3}
    GRASP_BACK = {"z": 0.05, "xy": 0.1}
    HAND_PALM_OFFSET = 0.05  # hand_palm_linkは指の付け根なので、把持のために少しずらす必要がある
    HAND_PALM_Z_OFFSET = 0.075
    DETECT_CNT = 10
    TROFAST_Y_OFFSET = 0.2

    def __init__(self):
        # 変数の初期化
        self.instruction_list = []
        self.detection_list   = []

        # configファイルの受信
        self.coordinates = self.load_json(self.get_path(["config", "coordinates.json"]))
        self.poses       = self.load_json(self.get_path(["config", "poses.json"]))

        # ROS通信関連の初期化
        tf_from_bbox_srv_name = "set_tf_from_bbox"
        rospy.wait_for_service(tf_from_bbox_srv_name)
        self.tf_from_bbox_clt = rospy.ServiceProxy(tf_from_bbox_srv_name, SetTransformFromBBox)

        obj_detection_name = "detection/get_object_detection"
        rospy.wait_for_service(obj_detection_name)
        self.detection_clt = rospy.ServiceProxy(obj_detection_name, GetObjectDetection)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.instruction_sub = rospy.Subscriber("/message",    String, self.instruction_cb, queue_size=10)
        self.detection_sub   = rospy.Subscriber("/detect_msg", String, self.detection_cb,   queue_size=10)

    @staticmethod
    def get_path(pathes, package="wrs_algorithm"):
        """
        ROSパッケージ名とファイルまでのパスを指定して、ファイルのパスを取得する
        """
        if not pathes:  # check if the list is empty
            rospy.logerr("Can NOT resolve file path.")
            raise ValueError("You must specify the path to file.")
        pkg_path = rospkg.RosPack().get_path(package)
        path = os.path.join(*pathes)
        return os.path.join(pkg_path, path)

    @staticmethod
    def load_json(path):
        """
        jsonファイルを辞書型で読み込む
        """
        with open(path, "r") as json_file:
            return json.load(json_file)

    def instruction_cb(self, msg):
        """
        指示文を受信する
        """
        rospy.loginfo("instruction received. [%s]", msg.data)
        self.instruction_list.append(msg.data)

    def detection_cb(self, msg):
        """
        検出結果を受信する
        """
        rospy.loginfo("received [Collision detected with %s]", msg.data)
        self.detection_list.append(msg.data)

    def get_relative_coordinate(self, parent, child):
        """
        tfで相対座標を取得する
        """
        try:
            # 4秒待機して各tfが存在すれば相対関係をセット
            trans = self.tf_buffer.lookup_transform(parent, child,rospy.Time.now(),rospy.Duration(4.0))
            return trans.transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            log_str = "failed to get transform between [{}] and [{}]\n".format(parent, child)
            log_str += traceback.format_exc()
            rospy.logerr(log_str)
            return None

    def goto_name(self, name):
        """
        waypoint名で指定された場所に移動する
        """
        if name in self.coordinates["positions"].keys():
            pos = self.coordinates["positions"][name]
            rospy.loginfo("go to [%s](%.2f, %.2f, %.2f)", name, pos[0], pos[1], pos[2])
            return omni_base.go_abs(pos[0], pos[1], pos[2])

        rospy.logerr("unknown waypoint name [%s]", name)
        return False

    def goto_pos(self, pos):
        """
        waypoint名で指定された場所に移動する
        """
        rospy.loginfo("go to [raw_pos](%.2f, %.2f, %.2f)", pos[0], pos[1], pos[2])
        return omni_base.go_abs(pos[0], pos[1], pos[2])

    def change_pose(self, name):
        """
        指定された姿勢名に遷移する
        """
        if name in self.poses.keys():
            rospy.loginfo("change pose to [%s]", name)
            return whole_body.move_to_joint_positions(self.poses[name])

        rospy.logerr("unknown pose name [%s]", name)
        return False

    def check_positions(self):
        """
        読み込んだ座標ファイルの座標を巡回する
        """
        whole_body.move_to_go()
        for wp_name in self.coordinates["routes"]["test"]:
            self.goto_name(wp_name)
            rospy.sleep(1)

    def get_latest_detection(self):
        """
        最新の認識結果が到着するまで待つ
        """
        res = self.detection_clt(GetObjectDetectionRequest())
        return res.bboxes

    def get_grasp_coordinate(self, bbox):
        """
        BBox情報から把持座標を取得する
        """
        # BBox情報からtfを生成して、座標を取得
        self.tf_from_bbox_clt.call(            SetTransformFromBBoxRequest(bbox=bbox, frame=self.GRASP_TF_NAME))
        rospy.sleep(1.0)  # tfが安定するのを待つ
        return self.get_relative_coordinate("map", self.GRASP_TF_NAME).translation

    @classmethod
    def get_most_graspable_bbox(cls, obj_list):
        """
        最も把持が行えそうなbboxを一つ返す。
        """
        # objが一つもない場合は、Noneを返す
        obj = cls.get_most_graspable_obj(obj_list)
        if obj is None: return None
        return obj["bbox"]

    @classmethod
    def get_most_graspable_obj(cls, obj_list):
        """
        把持すべきscoreが最も高い物体を返す。
        """
        extracted = []
        extract_str = "detected object list\n"
        ignore_str  = ""
        for obj in obj_list:
            info_str = "{:<15}({:.2%}, {:3d}, {:3d}, {:3d}, {:3d})\n".format(obj.label, obj.score, obj.x, obj.y, obj.w, obj.h)
            if obj.label in cls.IGNORE_LIST:
                ignore_str += "- ignored  : " + info_str
            else:
                score = cls.calc_score_bbox(obj)
                extracted.append({"bbox": obj, "score": score, "label": obj.label})
                extract_str += "- extracted: {:07.3f} ".format(score) + info_str

        rospy.loginfo(extract_str + ignore_str)

        # つかむべきかのscoreが一番高い物体を返す
        for obj_info in sorted(extracted, key=lambda x: x["score"], reverse=True):
            obj      = obj_info["bbox"]
            info_str = "{} ({:.2%}, {:3d}, {:3d}, {:3d}, {:3d})\n".format(obj.label, obj.score, obj.x, obj.y, obj.w, obj.h )
            rospy.loginfo("selected bbox: " + info_str)
            return obj_info

        # objが一つもない場合は、Noneを返す
        return None

    @classmethod
    def calc_score_bbox(cls, bbox):
        """
        detector_msgs/BBoxのスコアを計算する
        """
        gravity_x = bbox.x + bbox.w / 2
        gravity_y = bbox.y + bbox.h / 2
        xy_diff   = abs(320- gravity_x) / 320 + abs(360 - gravity_y) / 240

        return 1 / xy_diff

    @classmethod
    def get_most_graspable_bboxes_by_label(cls, obj_list, label):
        """
        label名が一致するオブジェクトの中から最も把持すべき物体のbboxを返す
        """
        match_objs = [obj for obj in obj_list if obj.label in label]
        if not match_objs:
            rospy.logwarn("Cannot find a object which labeled with similar name.")
            return None
        return cls.get_most_graspable_bbox(match_objs)

    @staticmethod
    def extract_target_obj_and_person(instruction):
        """
        指示文から対象となる物体名称を抽出する
        """
        #TODO: 関数は未完成です。引数のinstructionを利用すること
        rospy.loginfo("[extract_target_obj_and_person] instruction:"+  instruction)
        target_obj    = "apple"
        target_person = "right"

        return target_obj, target_person

    def grasp_from_side(self, pos_x, pos_y, pos_z, yaw, pitch, roll, preliminary="-y"):
        """
        把持の一連の動作を行う

        NOTE: tall_tableに対しての予備動作を生成するときはpreliminary="-y"と設定することになる。
        """
        if preliminary not in [ "+y", "-y", "+x", "-x" ]: raise RuntimeError("unnkown graps preliminary type [{}]".format(preliminary))

        rospy.loginfo("move hand to grasp (%.2f, %.2f, %.2f)", pos_x, pos_y, pos_z)

        grasp_back_safe = {"x": pos_x, "y": pos_y, "z": pos_z + self.GRASP_BACK["z"]}
        grasp_back = {"x": pos_x, "y": pos_y, "z": pos_z + self.GRASP_BACK["z"]}
        grasp_pos = {"x": pos_x, "y": pos_y, "z": pos_z}

        if "+" in preliminary:
            sign = 1
        elif "-" in preliminary:
            sign = -1

        if "x" in preliminary:
            grasp_back_safe["x"] += sign * self.GRASP_BACK_SAFE["xy"]
            grasp_back["x"] += sign * self.GRASP_BACK["xy"]
        elif "y" in preliminary:
            grasp_back_safe["y"] += sign * self.GRASP_BACK_SAFE["xy"]
            grasp_back["y"] += sign * self.GRASP_BACK["xy"]

        gripper.command(1)
        whole_body.move_end_effector_pose(grasp_back_safe["x"], grasp_back_safe["y"], grasp_back_safe["z"], yaw, pitch, roll)
        whole_body.move_end_effector_pose( grasp_back["x"], grasp_back["y"], grasp_back["z"], yaw, pitch, roll)
        whole_body.move_end_effector_pose(
            grasp_pos["x"], grasp_pos["y"], grasp_pos["z"], yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(grasp_back_safe["x"], grasp_back_safe["y"], grasp_back_safe["z"], yaw, pitch, roll)

    def grasp_from_front_side(self, grasp_pos):
        """
        正面把持を行う
        ややアームを下に向けている
        """
        grasp_pos.y -= self.HAND_PALM_OFFSET
        rospy.loginfo("grasp_from_front_side (%.2f, %.2f, %.2f)",grasp_pos.x, grasp_pos.y, grasp_pos.z)
        self.grasp_from_side(grasp_pos.x, grasp_pos.y, grasp_pos.z, -90, -100, 0, "-y")

    def grasp_from_upper_side(self, grasp_pos):
        """
        上面から把持を行う
        オブジェクトに寄るときは、y軸から近づく上面からは近づかない
        """
        grasp_pos.z += self.HAND_PALM_Z_OFFSET
        rospy.loginfo("grasp_from_upper_side (%.2f, %.2f, %.2f)",grasp_pos.x, grasp_pos.y, grasp_pos.z)
        self.grasp_from_side(grasp_pos.x, grasp_pos.y, grasp_pos.z, -90, -160, 0, "-y")

    def exec_graspable_method(self, grasp_pos, label=""):
        """
        task1専用:posの位置によって把持方法を判定し実行する。
        """
        method = None
        graspable_y = 1.85  # これ以上奥は把持できない
        desk_y = 1.5
        desk_z = 0.35

        # 把持禁止判定
        if graspable_y < grasp_pos.y and desk_z > grasp_pos.z:
            return False

        if label in ["cup", "frisbee", "bowl"]:
            # bowlの張り付き対策
            method = self.grasp_from_upper_side
        else:
            if desk_y < grasp_pos.y and desk_z > grasp_pos.z:
                # 机の下である場合
                method = self.grasp_from_front_side
            else:
                method = self.grasp_from_upper_side

        method(grasp_pos)
        return True

    def put_in_place(self, place, into_pose):
        # 指定場所に入れ、all_neutral姿勢を取る。
        self.change_pose("look_at_near_floor")
        a = "go_palce" # TODO 不要な変数
        self.goto_name(place)
        self.change_pose("all_neutral")
        self.change_pose(into_pose)
        gripper.command(1)
        rospy.sleep(5.0)
        self.change_pose("all_neutral")

    def pull_out_trofast(self, x, y, z, yaw, pitch, roll):
        # trofastの引き出しを引き出す
        self.goto_name("stair_like_drawer")
        self.change_pose("grasp_on_table")
        a = True  # TODO 不要な変数
        gripper.command(1)
        whole_body.move_end_effector_pose(x, y + self.TROFAST_Y_OFFSET, z, yaw, pitch, roll)
        whole_body.move_end_effector_pose(x, y, z, yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(x, y + self.TROFAST_Y_OFFSET, z, yaw,  pitch, roll)
        gripper.command(1)
        self.change_pose("all_neutral")

    def push_in_trofast(self, pos_x, pos_y, pos_z, yaw, pitch, roll):
        """
        trofastの引き出しを戻す
        NOTE:サンプル
            self.push_in_trofast(0.178, -0.29, 0.75, -90, 100, 0)
        """
        self.goto_name("stair_like_drawer")
        self.change_pose("grasp_on_table")
        pos_y+=self.HAND_PALM_OFFSET

        # 予備動作-押し込む
        whole_body.move_end_effector_pose( pos_x, pos_y +    self.TROFAST_Y_OFFSET * 1.5, pos_z, yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(  pos_x, pos_y + self.TROFAST_Y_OFFSET, pos_z, yaw, pitch, roll)
        whole_body.move_end_effector_pose(            pos_x, pos_y, pos_z, yaw, pitch, roll)

        self.change_pose("all_neutral")

    def deliver_to_target(self, target_obj, target_person):
        """
        棚で取得したものを人に渡す。
        """
        self.change_pose("look_at_near_floor")
        self.goto_name("shelf")
        self.change_pose("look_at_shelf")

        rospy.loginfo("target_obj: " + target_obj + "  target_person: " + target_person)
        # 物体検出結果から、把持するbboxを決定
        detected_objs = self.get_latest_detection()
        grasp_bbox = self.get_most_graspable_bboxes_by_label(detected_objs.bboxes, target_obj)
        if grasp_bbox is None:
            rospy.logwarn("Cannot find object to grasp. task2b is aborted.")
            return

        # BBoxの3次元座標を取得して、その座標で把持する
        grasp_pos = self.get_grasp_coordinate(grasp_bbox)
        self.change_pose("grasp_on_shelf")
        self.grasp_from_front_side(grasp_pos)
        self.change_pose("all_neutral")

        # target_personの前に持っていく
        self.change_pose("look_at_near_floor")
        self.goto_name("person_b")    # TODO: 配達先が固定されているので修正
        self.change_pose("deliver_to_human")
        rospy.sleep(10.0)
        gripper.command(1)
        self.change_pose("all_neutral")

    def execute_avoid_blocks(self):
        # blockを避ける
        for i in range(3):
            detected_objs = self.get_latest_detection()
            bboxes = detected_objs.bboxes
            pos_bboxes = [get_grasp_coordinaselfte(bbox) for bbox in bboxes]
            waypoint = self.select_next_waypoint(i, pos_bboxes)
            # TODO メッセージを確認するためコメントアウトを外す
            # rospy.loginfo(waypoint)
            self.goto_pos(waypoint)

    def select_next_waypoint(self, current_stp, pos_bboxes):
        """
        waypoints から近い場所にあるものを除外し、最適なwaypointを返す。
        x座標を原点に近い方からxa,xb,xcに定義する。bboxを判断基準として移動先を決定する(デフォルトは0.45間隔)
        pos_bboxesは get_grasp_coordinate() 済みであること
        """
        interval = 0.45
        pos_xa = 1.7
        pos_xb = pos_xa + interval
        pos_xc = pos_xb + interval

        # xa配列はcurrent_stpに関係している
        waypoints = {"xa": [ [pos_xa, 2.5, 45],[pos_xa, 2.9, 45],[pos_xa, 3.3, 90] ], 
                     "xb": [ [pos_xb, 2.5, 90], [pos_xb, 2.9, 90], [pos_xb, 3.3, 90] ],
            "xc": [ [pos_xc, 2.5, 135],   [pos_xc, 2.9, 135],  [pos_xc, 3.3, 90 ]]
        }

        # posがxa,xb,xcのラインに近い場合は候補から削除
        is_to_xa = True
        is_to_xb = True
        is_to_xc = True
        for bbox in pos_bboxes:
            pos_x = bbox.x
            # TODO デバッグ時にコメントアウトを外す
            # rospy.loginfo("detected object obj.x = {:.2f}".format(bbox.x))

            # NOTE Hint:ｙ座標次第で無視してよいオブジェクトもある。
            if pos_x < pos_xa + (interval/2):
                is_to_xa = False
                # rospy.loginfo("is_to_xa=False")
                continue
            elif pos_x < pos_xb + (interval/2):
                is_to_xb = False
                # rospy.loginfo("is_to_xb=False")
                continue
            elif pos_x < pos_xc + (interval/2):
                is_to_xc = False
                # rospy.loginfo("is_to_xc=False")
                continue

        x_line = None   # xa,xb,xcいずれかのリストが入る
        # NOTE 優先的にxcに移動する
        if is_to_xc:
            x_line = waypoints["xc"]
            rospy.loginfo("select next waypoint_xc")
        elif is_to_xb:
            x_line = waypoints["xb"]
            rospy.loginfo("select next waypoint_xb")
        elif is_to_xa:
            x_line = waypoints["xa"]
            rospy.loginfo("select next waypoint_xa")
        else:
            # a,b,cいずれにも移動できない場合
            x_line = waypoints["xb"]
            rospy.loginfo("select default waypoint")

        return x_line[current_stp]

    def identify_object_category(self, label):
        """ 62203145 宇野友貴 """

        categories = {
            # Shape items
            "baseball": "Shape items",
            "tennis ball": "Shape items",
            "mini soccer ball": "Shape items",
            "soft ball": "Shape items",
            "racquetball": "Shape items",
            "golf ball": "Shape items",
            "marbles": "Shape items",
            "foam brick": "Shape items",
            "dice": "Shape items",
            
            # Tools
            "large marker": "Tools",
            "small marker": "Tools",
            "keys": "Tools",
            "bolt": "Tools",
            "nut": "Tools",
            "clamps": "Tools",
            
            # Food
            "cheez-it cracker box": "Food",
            "domino sugar box": "Food",
            "jell-o chocolate pudding box": "Food",
            "jell-o strawberry gelatin box": "Food",
            "spam potted meat can": "Food",
            "master chef coffee can": "Food",
            "starkist tuna fish can": "Food",
            "pringles chips can": "Food",
            "french's mustard bottle": "Food",
            "tomato soup can": "Food",
            "plastic banana": "Food",
            "plastic strawberry": "Food",
            "plastic apple": "Food",
            "plastic lemon": "Food",
            "plastic peach": "Food",
            "plastic pear": "Food",
            "plastic orange": "Food",
            "plastic plum": "Food",
            
            # Kitchen items
            "windex spray bottle": "Kitchen items",
            "scrub cleanser bottle": "Kitchen items",
            "scotch brite dobie sponge": "Kitchen items",
            "pitcher base": "Kitchen items",
            "pitcher lid": "Kitchen items",
            "plate": "Kitchen items",
            "bowl": "Kitchen items",
            "fork": "Kitchen items",
            "spoon": "Kitchen items",
            "spatula": "Kitchen items",
            "wine glass": "Kitchen items",
            "mug": "Kitchen items",
            
            # Orientation-based items
            "small marker": "Orientation-based items",
            "large marker": "Orientation-based items",
            "fork": "Orientation-based items",
            
            # Task items
            "rubik's cube": "Task items",
            "colored wood blocks": "Task items",
            "9-peg-hole test": "Task items",
            "toy airplane": "Task items",
            "lego duplo": "Task items",
            "magazine": "Task items",
            "black t-shirt": "Task items",
            "timer": "Task items",
        }
        return categories.get(label, "Unknown objects/Boss")

    def get_tray_for_category(self, category):
        """ 62203145 宇野友貴 """

        tray_mapping = {
            "Shape items": "drawer_left",
            "Tools": "drawer_top",
            "Food": "tray_a",
            "Kitchen items": "container_a",
            "Orientation-based items": "container_b",
            "Task items": "bin_a",
            "Unknown objects/Boss": "bin_b"
        }
        return tray_mapping.get(category, "bin_b")
    

    # def plan_safe_grasp(self, grasp_pos, detected_objs):
    #     # 周囲の物体を考慮して安全な把持計画を立てるメソッド
    #     for obj in detected_objs:
    #         if obj != grasp_pos:
    #             # print(f"#####################################\n {type(grasp_pos)}\n##################################")
    #             distance = self.calculate_distance(grasp_pos, obj)
    #             if distance < 0.1:  # 10cm以内に他の物体がある場合
    #                 # 把持位置や姿勢を調整
    #                 grasp_pos = self.adjust_grasp_position(grasp_pos, obj)
    #     return grasp_pos
    
    # def calculate_distance(self, pos1, pos2):
    #     # return int(math.sqrt(pos1.x ** 2 + pos1.y ** 2 + pos1.z **2) - math.sqrt(pos2.x ** 2 + pos2.y ** 2 + pos2.z ** 2))
    #     return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pos1, pos2)))

    def execute_task1(self):
        """ 62203145 宇野友貴 """
        """
        改善されたtask1の実行
        """
        rospy.loginfo("#### start Task 1 ####")
        hsr_positions = [
            ("tall_table", "look_at_tall_table"),
            ("near_long_table_l", "look_at_near_floor"),
            ("long_table_r", "look_at_tall_table"),
            ("tray")
        ]

        total_score = 0
        start_time = rospy.Time.now()

        while (rospy.Time.now() - start_time).to_sec() < 600:  # 10分間繰り返す
            for plc, pose in hsr_positions:
                self.goto_name(plc)
                self.change_pose(pose)
                gripper.command(0)

                detected_objs = self.get_latest_detection()
                graspable_obj = self.get_most_graspable_obj(detected_objs.bboxes)

                if graspable_obj is None:
                    rospy.logwarn("Cannot determine object to grasp. Moving to next position.")
                    continue

                label = graspable_obj["label"]
                grasp_bbox = graspable_obj["bbox"]
                # ラベルの確認
                rospy.loginfo("grasp the " + label)

                grasp_pos = self.get_grasp_coordinate(grasp_bbox)
                # safe_grasp_pos = self.plan_safe_grasp(grasp_pos, detected_objs.bboxes)

                self.change_pose("grasp_on_table")
                if self.exec_graspable_method(grasp_pos, label):
                    self.change_pose("all_neutral")

                    category = self.identify_object_category(label)
                    tray = self.get_tray_for_category(category)

                    self.put_in_place(tray, self.coordinates["positions"][tray])

                    # if tray != "bin_b_place":
                    #     total_score += 10  # カテゴリが正解

                    rospy.loginfo(f"Current score: {total_score}")

                    # if total_score >= 120:
                    #     rospy.loginfo("Target score reached! Ending Task 1.")
                    #     return

        rospy.loginfo(f"Task 1 completed. Final score: {total_score}")

    def execute_task2a(self):
        """
        task2aを実行する
        """
        rospy.loginfo("#### start Task 2a ####")
        self.change_pose("look_at_near_floor")
        gripper.command(0)
        self.change_pose("look_at_near_floor")
        self.goto_name("standby_2a")

        # 落ちているブロックを避けて移動
        self.execute_avoid_blocks()

        self.goto_name("go_throw_2a")
        whole_body.move_to_go()

    def execute_task2b(self):
        """
        task2bを実行する
        """
        rospy.loginfo("#### start Task 2b ####")

        # 命令文を取得
        if self.instruction_list:
            latest_instruction = self.instruction_list[-1]
            rospy.loginfo("recieved instruction: %s", latest_instruction)
        else:
            rospy.logwarn("instruction_list is None")
            return

        # 命令内容を解釈
        target_obj, target_person = self.extract_target_obj_and_person(latest_instruction)

        # 指定したオブジェクトを指定した配達先へ
        if target_obj and target_person:            self.deliver_to_target(target_obj, target_person)

    def run(self):
        """
        全てのタスクを実行する
        """
        self.change_pose("all_neutral")
        self.execute_task1()
        # self.execute_task2a()
        # self.execute_task2b()


def main():
    """
    WRS環境内でタスクを実行するためのメインノードを起動する
    """
    rospy.init_node('main_controller')
    try:
        ctrl = WrsMainController()
        rospy.loginfo("node initialized [%s]", rospy.get_name())

        # タスクの実行モードを確認する
        if rospy.get_param("~test_mode", default=False) is True:
            rospy.loginfo("#### start with TEST mode. ####")
            ctrl.check_positions()
        else:
            rospy.loginfo("#### start with NORMAL mode. ####")
            ctrl.run()

    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

