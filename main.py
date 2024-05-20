import cv2
import numpy as np
import math
import os

inputpath = "input/"
outputpath = "output/"

def run(img_path):
    raw_image = cv2.imread(img_path)
    clock_image = cv2.imread(img_path)

    def getBlur(img):
        # Chuyển ảnh màu sang ảnh HSV
        hsv = img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = cv2.bitwise_not(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        _, thresh = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(thresh, (5, 5), 0)

        return blur

    blur = getBlur(raw_image)

    # Tìm vùng hình tròn (đồng hồ)
    # Tạo maxRadius bằng 1.5 lần max(width, height) của ảnh
    max_radius = int(max(raw_image.shape) // 2)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=400, param1=50, param2=30, minRadius=50, maxRadius=max_radius)
    # Lấy hình tròn lớn nhất
    clock_circle = None
    center_x, center_y = None, None

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Sắp xếp các hình tròn theo thứ tự giảm dần của bán kính
        circles = sorted(circles[0], key=lambda x: x[2], reverse=True)
        max_circle = circles[0]

        # Vẽ hình tròn lớn nhất = màu đỏ
        clock_circle = max_circle
        center_x, center_y = max_circle[0], max_circle[1]
        copy_image = raw_image.copy()
        cv2.circle(copy_image, (max_circle[0], max_circle[1]), max_circle[2], (0, 0, 255), 2)
        
        # # Hiển thị ảnh
        # cv2.imshow('Clock Image', copy_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("Không tìm thấy hình tròn trong ảnh.")
        exit()

    # Tạo ảnh mới bằng cách cắt ảnh gốc theo hình tròn lớn nhất
    cut_image = None
    image_show = None
    lines = []
    if clock_circle is not None:
        # cắt ảnh theo hình tròn lớn nhất
        x, y, r = clock_circle
        # bán kính tràn ra ngoài ảnh thì lấy min(x, y)
        if(r > x or r > y):
            r = min(x, y)
        clock_image = raw_image[y - r:y + r, x - r:x + r]
        
        # resize ảnh về kích thước 500x500
        height, width = clock_image.shape[:2]
        scale = 500 / max(height, width)
        clock_image = cv2.resize(clock_image, None, fx=scale, fy=scale)
        cut_image = clock_image.copy()

        # Chuyển ảnh màu sang ảnh xám
        clock_image = getBlur(clock_image)
        
        # Tìm cạnh của ảnh
        edges = cv2.Canny(clock_image, 50, 170, apertureSize=3)

        # Sử dụng Hough Line Transform để tìm đường thẳng
        # Tạo maxLineGap từ bán kính của ảnh
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=10, maxLineGap=1)
        # image_show = cut_image.copy()
    else:
        print("Không tìm thấy hình tròn trong ảnh.")
        exit()
    if lines is not None:
        # Vẽ đường thẳng
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # cv2.line(image_show, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # print(f"lines: {len(lines)}")
    # cv2.imshow('Default Lines', image_show)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Tìm các đường gần tâm đường tròn gộp chúng lại
    # Tính tâm của đường tròn theo width và height ban đầu là 500
    center = (250, 250)

    # Tìm các đường có 1 hướng vector chỉ về tâm của đường tròn
    image_show = cut_image.copy()
    focus_center_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Tính tâm của đoạn thẳng
        line_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        # Tính vector từ tâm của đường tròn tới tâm của đoạn thẳng
        vector = (line_center[0] - center[0], line_center[1] - center[1])
        # Tính vector của đoạn thẳng
        line_vector = (x2 - x1, y2 - y1)
        # Tính góc giữa 2 vector
        dot = vector[0] * line_vector[0] + vector[1] * line_vector[1]
        length_vector = np.linalg.norm(vector)
        length_line_vector = np.linalg.norm(line_vector)
        cos_theta = dot / (length_vector * length_line_vector)
        theta = np.arccos(cos_theta)
        theta = np.degrees(theta)
        # Nếu góc giữa 2 vector nhỏ hơn 10 độ thì đường thẳng này gần tâm
        if theta < 20 or theta > 160:
            focus_center_lines.append(line)
            # cv2.line(image_show, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # print(f"Số lượng trỏ vào tâm: {len(focus_center_lines)}")

    # cv2.imshow('Lines Near Center', image_show)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Nhóm các đường thẳng gần song song lại
    groups = []
    for line in focus_center_lines:
        x1, y1, x2, y2 = line[0]

        # Kiểm tra xem đường thẳng có nằm trong bán kính không
        if np.linalg.norm(np.array([x1, y1]) - np.array(center)) > 250 or np.linalg.norm(np.array([x2, y2]) - np.array(center)) > 250:
            continue
        # Tính góc của đường thẳng
        angle = np.arctan2(y2 - y1, x2 - x1)
        # Tính góc của đường thẳng so tâm của đường tròn
        angle = np.degrees(angle)
        found = False
        for group in groups:
            if abs(angle - group[0]) < 12 or abs(angle - group[0] - 180) < 12 or abs(angle - group[0] + 180) < 12:
                group.append(line)
                found = True
                break
        if not found:
            groups.append([angle, line])

    # Sắp xếp các nhóm theo số lượng đường thẳng giảm dần
    groups = sorted(groups, key=lambda x: len(x), reverse=True)
    # Lấy 3 nhóm có nhiều đường thẳng nhất
    groups = groups[:3]
    if len(groups) != 3:
        print("Không tìm được 3 kim đồng hồ.")
        exit()

    # print(f"groups: {groups}")

    # Vẽ mỗi group 1 màu ngẫu nhiên
    image_show = cut_image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for i, group in enumerate(groups):
        for line in group[1:]:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_show, (x1, y1), (x2, y2), colors[i], 2)

    # cv2.imshow('Group Lines', image_show)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Loại bỏ các đường thẳng quá xa so với các đường thẳng cùng nhóm
    for group in groups:
        # Lấy tâm đồng hồ (250, 250) làm điểm trung tâm, nhóm các đường thẳng theo 2 nhóm trên và dưới tâm, loại bỏ nhóm ít đường thẳng
        # Tính khoảng cách và góc của các đường thẳng so với tâm đồng hồ
        distances = []
        angles = []
        for line in group[1:]:
            x1, y1, x2, y2 = line[0]
            # Tính tâm của đoạn thẳng
            line_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            # Tính vector từ tâm của đường tròn tới tâm của đoạn thẳng
            vector = (line_center[0] - center[0], line_center[1] - center[1])
            # Tính vector của đoạn thẳng (Xuất phát từ tâm đồng hồ)
            if abs(250 - x1 - y1) < abs( 250 - x2 - y2):
                line_vector = (x2 - x1, y2 - y1)
            else:
                line_vector = (x1 - x2, y1 - y2)
            # Tính góc giữa 2 vector
            dot = vector[0] * line_vector[0] + vector[1] * line_vector[1]
            length_vector = np.linalg.norm(vector)
            length_line_vector = np.linalg.norm(line_vector)
            cos_theta = dot / (length_vector * length_line_vector)
            theta = np.arccos(cos_theta)
            theta = np.degrees(theta)
            distances.append(length_vector)
            angles.append(theta)
        # Nhóm các đường thẳng theo góc
        groups_temp = []
        for i in range(len(group[1:])):
            found = False
            for group_temp in groups_temp:
                if abs(angles[i] - group_temp[0]) < 30 :
                    group_temp.append(group[1:][i])
                    found = True
                    break
            if not found:
                groups_temp.append([angles[i], group[1:][i]])
        # Sắp xếp các nhóm theo độ dài trung bình của các đường thẳng trong nhóm
        # print(f"groups_temp: {groups_temp}")
        for i in range(len(groups_temp)):
            group_temp = groups_temp[i]
            distances_temp = [np.linalg.norm(np.array(line[0][:2]) - np.array(line[0][2:])) for line in group_temp[1:]]
            # Thêm vào đầu mảng
            groups_temp[i].insert(0, np.mean(distances_temp))
        groups_temp = sorted(groups_temp, key=lambda x: x[0], reverse=True)

        # Bỏ dinstance temp
        for i in range(len(groups_temp)):
            groups_temp[i] = groups_temp[i][1:]
            
        # Lấy nhóm có độ dài trung bình lớn nhất
        group[1:] = groups_temp[0][1:]

    # # Vẽ mỗi group 1 màu ngẫu nhiên
    # image_show = cut_image.copy()
    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # for i, group in enumerate(groups):
    #     for line in group[1:]:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(image_show, (x1, y1), (x2, y2), colors[i], 2)

    # cv2.imshow('Group Linesaa', image_show)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Tìm 3 kim đồng hồ
    def distance_between_parallel_lines(line1, line2):
        # Get the coordinates of two points on each line
        x1_1, y1_1, x2_1, y2_1 = line1[0]
        x1_2, y1_2, x2_2, y2_2 = line2[0]

        # Create two direction vectors of two straight lines
        vector1 = np.array([x2_1 - x1_1, y2_1 - y1_1])
        vector2 = np.array([x2_2 - x1_2, y2_2 - y1_2])

        #Creates a vector connecting a point on one line to a point on the other line
        vector_between_lines = np.array([x1_2 - x1_1, y1_2 - y1_1])

        #Calculates the perpendicular distance between the two lines.
        distance = np.abs(np.cross(vector1, vector_between_lines)) / np.linalg.norm(vector1)

        return distance

    clock_hands = []
    for group in groups:
        hand = None

        # Tìm điểm gần tâm nhất và xa tâm nhất trong các điểm của các đường thẳng
        min_point = None
        max_point = None
        
        # Tìm độ dày của kim đồng hồ (tính bằng khoảng cách giữa tâm của 2 đường thẳng) < max_length
        points = []
        for line in group[1:]:
            x1, y1, x2, y2 = line[0]
            if min_point is None:
                min_point = (x1, y1)
            if max_point is None:
                max_point = (x1, y1)
            if np.linalg.norm(np.array([x1, y1]) - np.array(center)) < np.linalg.norm(np.array(min_point) - np.array(center)):
                min_point = (x1, y1)
            if np.linalg.norm(np.array([x2, y2]) - np.array(center)) < np.linalg.norm(np.array(min_point) - np.array(center)):
                min_point = (x2, y2)
            if np.linalg.norm(np.array([x2, y2]) - np.array(center)) > np.linalg.norm(np.array(max_point) - np.array(center)):
                max_point = (x2, y2)
            if np.linalg.norm(np.array([x1, y1]) - np.array(center)) > np.linalg.norm(np.array(max_point) - np.array(center)):
                max_point = (x1, y1)


        # Tính độ dày của kim đồng hồ = độ dài lớn nhất giữa 2 điểm
        thickness = 0
        for i in range(len(group[1:])):
            for j in range(i + 1, len(group[1:])):
                distance = distance_between_parallel_lines(group[1:][i], group[1:][j])
                if distance > thickness:
                    thickness = distance
            
        # Tạo đường thẳng từ min_point đến max_point
        hand = (250, 250, max_point[0], max_point[1])
        clock_hands.append((hand, thickness))

    # for hand in clock_hands:
    #     print(f"thickness: {hand[1]}")

    # print(f"kim đồng hồ: {(clock_hands)}")

    # # Vẽ các kim đồng hồ trên ảnh
    # clock_image = cut_image.copy()
    # for line in clock_hands:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(clock_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
    # # Hiển thị ảnh với các kim đồng hồ
    # cv2.imshow('3 hand clock', clock_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Tìm ra kim giờ, kim phút và kim giây
    hour_hand = None
    minute_hand = None
    second_hand = None

    # Kim giây là kim có thickness nhỏ nhất
    clock_hands = sorted(clock_hands, key=lambda x: x[1])
    second_hand = clock_hands[0][0]
    clock_hands = clock_hands[1:]

    # Kim giờ là ngắn nhất trong 2 kim còn lại
    # Sắp xếp các kim còn lại theo chiều dài tăng dần
    clock_hands = sorted(clock_hands, key=lambda x: np.linalg.norm(np.array(x[0][:2]) - np.array(x[0][2:])))
    hour_hand = clock_hands[0][0]
    minute_hand = clock_hands[1][0]

    # # Vẽ kim giờ màu đỏ, kim phút xanh dương và kim giây xanh lá
    # clock_image = cut_image.copy()
    # x1, y1, x2, y2 = hour_hand
    # cv2.line(clock_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # x1, y1, x2, y2 = minute_hand
    # cv2.line(clock_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # x1, y1, x2, y2 = second_hand
    # cv2.line(clock_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # # Hiển thị ảnh với các kim đồng hồ
    # cv2.imshow('Clock Hands', clock_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Vẽ hình chữ nhật bao quanh kim giờ, kim phút và kim giây
    clock_image = cut_image.copy()
    x1, y1, x2, y2 = hour_hand
    cv2.rectangle(clock_image, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), (0, 0, 255), 2)
    x1, y1, x2, y2 = minute_hand
    cv2.rectangle(clock_image, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), (255, 0, 0), 2)
    x1, y1, x2, y2 = second_hand
    cv2.rectangle(clock_image, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), (0, 255, 0), 2)

    # # Hiển thị ảnh với các kim đồng hồ
    # cv2.imshow('Clock Hands', clock_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Tính góc cho các kim đồng hồ dựa theo tâm của đồng hồ
    # Tính góc của kim so với vector chỉ hướng lên trên
    def get_angle(hand):
        # u là vector của kim đồng hồ
        x1, y1, x2, y2 = hand
        # Sắp xếp lại để vector đi từ tâm đồng hồ (250, 250)
        # Tìm xem đầu nào gần tâm đồng hồ nhất
        if np.linalg.norm(np.array([x1, y1]) - np.array([250, 250])) < np.linalg.norm(np.array([x2, y2]) - np.array([250, 250])):
            x1, y1, x2, y2 = x2, y2, x1, y1
        u = [x2 - x1, y2 - y1]

        # v là vector chỉ hướng lên trên
        v = [0, 100]
        # Tính tích vô hướng của 2 vector
        dot_uv = u[0] * v[0] + u[1] * v[1]

        # Tính độ dài của 2 vector
        length_u = math.sqrt(u[0]**2 + u[1]**2)
        length_v = math.sqrt(v[0]**2 + v[1]**2)

        # Tính cos của góc giữa 2 vector u.v / (|u| * |v|)
        cos_theta = dot_uv / (length_u * length_v)

        # Đảm bảo cos_theta nằm trong khoảng [-1, 1]
        cos_theta = max(min(cos_theta, 1.0), -1.0)

        # Tính góc giữa 2 vector
        theta = math.acos(cos_theta)

        # Chuyển radian sang độ
        theta_degrees = math.degrees(theta)

        # Tính tích có hướng của 2 vector
        cross_uv = u[0] * v[1] - u[1] * v[0]

        # Nếu tích có hướng > 0 thì góc là góc bù của theta
        if cross_uv > 0:
            # Trả về góc bù của theta
            return 360 - theta_degrees
        else:
            return theta_degrees

    # Tính góc cho các kim đồng hồ
    hour_angle = get_angle(hour_hand)
    minute_angle = get_angle(minute_hand)
    second_angle = get_angle(second_hand)

    # print(f"hour_angle: {hour_angle}")
    # print(f"minute_angle: {minute_angle}")
    # print(f"second_angle: {second_angle}")

    # Tính thời gian của đồng hồ
    hour_time = hour_angle / 30
    minute_time = minute_angle / 6
    second_time = second_angle / 6

    rounded_hour = round(hour_time)
    rounded_minute = round(minute_time)
    rounded_second = round(second_time)

    # nếu phút hoặc giây bằng 60 thì làm tròn thành 0
    if rounded_minute == 60:
        rounded_minute = 0
    if rounded_second == 60:
        rounded_second = 0 

    print(f"{img_path} Thời gian:{rounded_hour}:{rounded_minute}:{rounded_second}")

    # Viết thời gian lên ảnh
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(clock_image, f"{rounded_hour}:{rounded_minute}:{rounded_second}", (10, 30), font, 1, (0, 0, 255), 2)
    # # Hiển thị ảnh với thời gian
    # cv2.imshow('Clock Hands', clock_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Lưu ảnh với thời gian vào thư mục output
    output_filename = os.path.basename(img_path)
    output_path = os.path.join(outputpath, output_filename)
    cv2.imwrite(output_path, clock_image)

def main():
    # Đọc 10 ảnh từ thư mục input
    print("START")
    for i in range(1, 11):
        filename = f'clock{i}.jpg'
        img_path = os.path.join(inputpath, filename)
        if not os.path.exists(img_path):
            continue  
        run(img_path)
    print("DONE")
if __name__ == "__main__":
    main()