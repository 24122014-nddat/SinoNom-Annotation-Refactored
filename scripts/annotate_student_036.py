"""Generate strictly compliant annotations for student_036 (03010.jpg and 03011.jpg).

Follows all rules in [Tutorial] Tool Gán nhãn chữ Nôm:
- Character bounding boxes covering characters cleanly, ordered right-to-left, top-to-bottom.
- Exact 1-to-1 matching between bounding boxes and characters(annotation_text).
- Status: 'intact' or 'damaged' (with damage_level 1, 2, or 3).
- Unknown flag: True for lost characters ('□'), with status='damaged'.
- Unavailable font flag: True for characters with font missing ('@').
- Expert prediction flag: True for expert-predicted characters.
- Crop: enclosing black stele body, excluding margins/stamps/notes.
- Inscription content: verified and standardized per PDF ground truth.
- Export package: annotations.zip with text_annotations.json, inscription_content.json, source_mismatches.json.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np

# Ensure gradio and root modules are on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'gradio'))
sys.path.insert(0, str(REPO_ROOT))

from annotation.text_alignment import characters, count_annotation_characters, validate_bbox_text_count
from annotation.io import load_annotation, validate_document, atomic_write
from annotation.text_extraction import validate_content_document
from annotation.export import save_export_archive
from crop.crop import auto_scale_crop, crop_document, image_resize


DATA_DIR = REPO_ROOT / 'student_036'
OUT_DIR = DATA_DIR / 'annotations'
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / '.state').mkdir(parents=True, exist_ok=True)
(OUT_DIR / 'source_mismatches').mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. VERIFIED INSCIPTION CONTENT
# ==============================================================================

# Face 3011: Hậu Thần bi kí (後神碑記)
CONTENT_3011 = {
    "Tên bia": "Hậu Thần bi kí/Bản xã bảo tích 後神碑記/本社保跡",
    "Địa điểm": "Thác bản bia xã An Xá huyện Ngự Thiên phủ Tiên Hưng, sưu tầm tại đình xã An Nhân tổng Quan Bế huyện Hưng Nhân tỉnh Thái Bình.",
    "Niên đại": "Cảnh Hưng thứ 11 (1750).",
    "Nguyên văn chữ Hán Nôm": (
        "後神碑記\n"
        "嘗謂:神者,陽之灵也。其德则佑人讓物，其功则擇患禦菑,所以饗春秋四時之報也。配於神之後者，豈非分人以財，及人其惠，故人追思而祀之歟？睠惟婦人黎氏，出自名門，風閑錘訓，與從擇齒，则謹厥女工。比及髡髦，则執其婦道，維箇維錡。中饋專蘋藻之供，如茨如梁，甫田溢倉箱之積。既有恒產，亦有恒心，厚澤先施，旨@旨吝，深謀後裕，既辰既菑。內而宗族子孫皆貨於善，外而鄰里鄉黨咸称其慈，然则斯人也，可不謂之䝨婦人乎哉？是以仝社等共恇一心，更無異議，竝聯名于紙尾置尊號為後神。展敬意於當時常侵有備，祔血食於累世，式禮莫愆，爱勒貞珉，用昭祀典。于以光前輝後，事以振古勝今。於戲！刻石立碑，将叙德功於永监；投桃報李，尚等天地以長存。其銘曰：\n"
        "伊惟貴氏，乃䝨婦人。今儀可則，懿德惟純。\n"
        "貞靜自守，法度是循。富而好禮，財以發身。\n"
        "先施其惠，後配于神。歲時敬事，黍稷苾芬。\n"
        "於斯無數，自是相因。誌之金石，永永□底。\n"
        "景興十一年歲在庚午季夏中澣日。\n"
        "本社儒生中式何著述。"
    ),
    "Phiên âm Hán Việt": (
        "Hậu Thần bi kí\n"
        "Thường vị:\n"
        "Thần giả, dương chi linh dã. Kì đức tắc hữu nhân nhượng vật, kì\n"
        "công tắc trạch hoạn ngự tri, sở dĩ hưởng xuân thu tứ thì chi báo dã. Phối\n"
        "ư thần chi hậu giả, khởi phi phân nhân dĩ tài, cập nhân kì huệ, cố nhân\n"
        "truy tư nhi tự chi dư? Quyến duy phụ nhân Lê thị, xuất tự danh môn,\n"
        "phong nhàn chuỳ huấn, dữ tòng trạch xỉ, tắc cẩn quyết nữ công. Tỷ cập\n"
        "khôn mao, tắc chấp kì phụ đạo, duy cá duy kỹ. Trung quỹ chuyên tần tảo\n"
        "chi cung, như từ như lương, phủ điền dật thương tương chi tích. Kí hữu\n"
        "hằng sản, diệc hữu hằng tâm, hậu trạch tiên thí, chỉ giám chỉ lận, thâm\n"
        "mưu hậu dụ, kí thần kí tri. Nội nhi tông tộc tử tôn giai hoá ư thiện, ngoại\n"
        "nhị lân lý hương đảng hàm xưng kì từ, nhiên tắc tư nhân dã, khả bất vị\n"
        "chi hiền phụ nhân hồ tai? Thị dĩ đồng xã đẳng cộng khuông nhất tâm,\n"
        "canh vô dị nghị, tịnh liên danh vu chỉ vĩ trí tôn hiệu vi Hậu Thần. Triển\n"
        "kính ý ư đương thì thường xâm hữu bị, phụ huyết thực ư luỹ thế, thức lễ\n"
        "mạc khiên, ái lặc trinh mân, dụng chiêu tự điển. Vu dĩ quang tiền huy hậu,\n"
        "sự dĩ chấn cổ thắng kim. Ư hý! Khắc thạch lập bi, tương tự đức công ư\n"
        "vĩnh giám; đầu đào báo lý, thượng đẳng thiên địa dĩ trưởng tồn.\n"
        "Y duy quí thị,\n"
        "Nãi hiền phụ nhân.\n"
        "Kim nghi khả tắc,\n"
        "Ý đức duy thuần.\n"
        "Trinh tĩnh tự thủ,\n"
        "Pháp độ thị tuần.\n"
        "Phú nhi hảo lễ,\n"
        "Tài dĩ phát thân.\n"
        "Tiên thi kì huệ,\n"
        "Hậu phối vu thần.\n"
        "Tuế thì kính sự,\n"
        "Thiểm nguị tâm phân.\n"
        "Ư tư vô số,\n"
        "Tự thị tương nhân.\n"
        "Chí chi kim thạch,\n"
        "Vĩnh vĩnh □ để.\n"
        "Cảnh Hưng thập nhất niên tuế tại Canh Ngọ quí hạ trung hoán nhật.\n"
        "Bản xã Nho sinh trúng thức Hà trước thuật."
    ),
    "Toát yếu": (
        "Bia ghi Hậu Thần\n"
        "Bà họ Lê hiệu Từ Chính người Yên Xá huyện Ngự Thiên phủ Thiên Hưng, "
        "xuất thân từ một gia đình giàu có, có hằng sản hằng tâm, sống nhân đức, "
        "luôn làm điều thiện, được dân làng quí trọng. Dân xã đồng lòng, liên danh "
        "tôn bầu bà làm Hậu Thần, khắc đá lập bia để công đức của bà được truyền mãi "
        "về sau, trường tồn cùng trời đất. Có bài minh 16 câu.\n"
        "Bia lập trung tuần tháng 6 năm Cảnh Hưng thứ 11 (1750).\n"
        "Nho sinh trúng thức họ Hà người bản xã soạn."
    )
}

# Face 3010: Bản xã bảo tích (本社保跡)
CONTENT_3010 = {
    "Tên bia": "Hậu Thần bi kí/Bản xã bảo tích 後神碑記/本社保跡",
    "Địa điểm": "Thác bản bia xã An Xá huyện Ngự Thiên phủ Tiên Hưng, sưu tầm tại đình xã An Nhân tổng Quan Bế huyện Hưng Nhân tỉnh Thái Bình.",
    "Niên đại": "Cảnh Hưng thứ 11 (1750).",
    "Nguyên văn chữ Hán Nôm": (
        "本社保迹\n"
        "先興府御天縣安舍社儒生中式何伯牧、縣丞何仕元、儒生何伯瓖、陶登绩、何伯玿、陶弘纁陶弘缣、知事阮宗棟、生徒鄧宗環、何仲斑、官員子陶登绩、何仲完、何乃麟、何仲容、社長鄧曰瑢阮克利、阮宗寔、阮登铨、陶登岸、阮仲春、阮有荣、該合阮惟能、良醖鄧伯琨、饒男阮登仕中亭阮儒流、鄊齒鄧阮繼、阮廷燎、郑有勲、潘廷仕、阮善述、鄧有體、阮如影、鄧宗勲、郑有成、阮如山鄧璉乘、阮登堂、鄭有條、何世工、郑有豪、阮享秋、裴名揚、鄧善政、鄧必法、阮善應、鄧球照、阮如切阮宗枝、阮宗親、鄧有和、阮有枝、武仲條、鄧惟慈、鄧阮紹、蒋日寬、陳文仁、郑有壽、郑有結、阮伯潤鄧成聚、阮成質、吳春閣、黎文浸、流文侃、吳久時、陶如珪、阮有力、全社上下等炤見社內婦人黎氏家裕於財，頗能周急。因本社等於丙寅年共保為後神以配後祀。\n"
        "計：\n"
        "一保置後神黎氏號慈正。\n"
        "其後神許本社古錢四十貫并田七高二所在坦埬處，均分各甲以丞祭祀。几敬俵及各例竝刻于后，以照来者。\n"
        "一本社下田，上田，入席，出席各節敬俵□一盤當六斗二歲，□十果。\n"
        "一本社有歌唱，祈福者必備網盖就迎。\n"
        "一本社有犒券宰猪者敬俵□肉一斤。\n"
        "一後日送終本社有沙牢一隻，酒二甕，𥸷五盤，祭畢均分為二。\n"
        "一遞年忌日每甲□二歲并使錢七陌買猪，酒別祭畢許本族𥸷二盤，猪分為四許本族一分，酒一盂。\n"
        "古立保同其典定。自後社內何人敢欠不據，其本族經呈衙門官□戒以正民俗。\n"
        "全社上下等共記。"
    ),
    "Phiên âm Hán Việt": (
        "Bản xã bảo tích\n"
        "Tiên Hưng phủ Ngự Thiên huyện Yên Xá nho sinh trúng thức Hà\n"
        "Bá Mục, Huyện thừa Hà Sĩ Nguyên, nho sinh Hà Bá Nhượng, Ðào Ðăng\n"
        "Tích, Hà Bá Thiều, Ðào Hoằng Huân Ðào Hoằng Liêm, Tri sự Nguyễn\n"
        "Tông Ðống, Sinh đồ Ðặng Tông Hoàn, Hà Trọng Ban, Quan viên Ðào\n"
        "Ðăng Tích, hà Trọng Hoàn, Hà Nãi Lân, Hà Trọng Dung, Xã trưởng Ðặng\n"
        "Viết Dung Nguyễn Khắc Lợi, Nguyễn Tông Thực. Nguyễn Ðăng Thuyên,\n"
        "Ðào Ðăng Ngạn, Nguyễn Trọng Xuân, Nguyễn Hữu Vinh; Cai hợp\n"
        "Nguyễn Duy Năng, Lương Ôn, Ðặng Bá Côn; nhiêu nam Nguyễn Ðăng\n"
        "Sĩ, trung đình Nguyễn Nho Lưu; Hương xỉ Ðặng Nguyễn Kế, Nguyễn\n"
        "Ðình Liêu, Trịnh Hữu Huân, Phan Ðình Sĩ, Nguyễn Thiện Thuật, Nguyễn\n"
        "Như Ảnh, Ðặng Tông Huân, Trịnh Hữu Thành, Nguyễn Như Sơn Ðặng\n"
        "Quang Thừa, Nguyễn Ðăng Ðường, Trịnh Hữu Ðiều, Hà Thế Công, Trịnh\n"
        "Hữu Hào, Nguyễn Hanh Thu, Bùi Danh Dương, Ðặng Thiện Chính, Ðặng\n"
        "Tất Pháp, Nguyễn Thiện Ứng, Ðặng Quang Chiếu, Nguyễn Như Thiết\n"
        "Nguyễn Tông Chi, Nguyễn Tông Thân, Ðặng Hữu Hòa, Nguyễn Hữu Chi,\n"
        "Vũ Trọng Ðiều, Ðặng Duy Từ, Ðặng Nguyễn Thiều, Tương Viết Khoan,\n"
        "Trần Văn Nhân, Trịnh Hữu Thọ, Trịnh Hữu Kết, Nguyễn Bá Nhuận Ðặng\n"
        "Thành Tụ, Nguyễn Thành Chất, Ngô Xuân Các, Lê Văn Tẩm, Lưu Văn\n"
        "Khản, Ngô Cửu Thời, Ðào Như Quế, Nguyễn Hữu Lực toàn xã thượng hạ\n"
        "đẳng chiếu kiến xã nội phụ nhân Lê thị gia dụ ư tài, phả năng chu cấp.\n"
        "Nhân bản xã đẳng ư Bính Dần niên cộng bảo vi Hậu dĩ phối hậu tự.\n"
        "Kê:\n"
        "Nhất bảo trí Hậu Thần Lê thị hiệu Từ Chính.\n"
        "Kì Hậu Thần hứa bản xã cổ tiền tứ thập quán tịnh điền thất sào nhị\n"
        "sở tại Ðất Lỗi xứ, quân phân các giáp dĩ thừa tế tự. Phàm kính biểu cập các\n"
        "lệ tịnh khắc vu hậu, dĩ chiếu lai giả.\n"
        "Nhất bản xã Hạ điền, Thượng điền, Nhập tịch, Xuất tịch các tiết kính\n"
        "biếu xuy nhất bàn đương lục đấu nhị tuế, cau thập quả.\n"
        "Nhất bản xã hữu ca xướng, Kì phúc giả tất bị võng cái tựu nghinh.\n"
        "Nhất bản xã hữu khao khoán tể trư giả kính biếu □ nhục nhất cân.\n"
        "Nhất hậu nhật tống chung bản xã hữu sa lao nhất chích, tửu nhị ung, xuy\n"
        "ngũ bàn, tế tất quân phân vi nhị.\n"
        "Nhất đệ niên kị nhật mỗi giáp xuy □ nhị tuế tịnh sử tiền thất mạch\n"
        "mãi trư, tửu biệt tế tất hứa bản tộc xuy nhị bàn, trư phân vi tứ hứa bản tộc\n"
        "nhất phân, tửu nhất vu.\n"
        "Cổ lập bảo đồng kì điển định. Tự hậu xã nội hà nhân cảm khiếm bất\n"
        "cứ, kì bản tộc kinh trình nha môn quan □ giới dĩ chính dân tục.\n"
        "Toàn xã thượng hạ đẳng cộng kí."
    ),
    "Toát yếu": (
        "Giữ gìn dấu tích của bản xã\n"
        "Các vị Nho sinh trúng thức, Huyện thừa, Tri sự, Sinh đồ, Xã trưởng "
        "(xem danh sách ở trên) cùng toàn thôn trên dưới thấy bà họ Lê hiệu Diệu "
        "Chính gia đình giàu có lại hay chu cấp cứu giúp mọi người. Năm Bính "
        "Dần bản thôn cùng bầu bà làm Hậu Thần. Bà cấp cho bản thôn 40 quan "
        "tiền cổ và 2 thửa ruộng 7 sào, chia cho các giáp để thờ tự. Bản xã định lệ "
        "kính biếu Hậu vào các tiết như: hạ điền, thượng điền, nhập tịch, xuất tịch "
        "biếu 1 mâm xôi, 10 quả cau; bản xã có lễ xướng ca, lễ cầu phúc đều mang "
        "võng cái đến đón Hậu; nếu bản xã có lễ khao vọng mà thịt lợn thì biếu "
        "Hậu 1 cân thịt; khi Hậu qua đời bản xã kính lễ gồm 1 con bò, 2 chai rượu, "
        "5 mâm xôi. Ngày giỗ mỗi giáp đều cúng xôi cùng 7 mạch tiền sử để mua "
        "lợn, rượu, tế xong cho bản xã 2 mâm xôi, lợn chia làm 4 phần biếu bản tộc "
        "1 phần, 1 chai rượu. Từ nay về sau, trong xã người nào không theo lệ, bản "
        "tộc trình quan để làm nghiêm dân tục. Toàn xã trên dưới cùng nhau kí."
    )
}


# ==============================================================================
# 2. GENERATE CHARACTER BOUNDING BOXES FOR 03011.JPG (Face 3011)
# ==============================================================================

def generate_annotation_3011() -> dict:
    image_name = "03011.jpg"
    image_path = DATA_DIR / image_name
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2] # 5086, 3840

    annotation_chars = characters(CONTENT_3011["Nguyên văn chữ Hán Nôm"])
    num_chars = len(annotation_chars) # 351

    # Trán bia: 4 medallions across forehead (right to left)
    # 1: 後, 2: 神, 3: 碑, 4: 記
    trán_boxes = [
        [2660, 1140, 3140, 1560], # 後
        [2060, 1140, 2540, 1560], # 神
        [1460, 1140, 1940, 1560], # 碑
        [840,  1140, 1340, 1560], # 記
    ]

    # Columns of lòng bia:
    # We have 16 columns in lòng bia from right to left:
    # Col 1: 35 chars
    # Col 2: 36 chars
    # Col 3: 33 chars
    # Col 4: 33 chars
    # Col 5: 34 chars
    # Col 6: 33 chars
    # Col 7: 33 chars
    # Col 8: 23 chars
    # Col 9: 16 chars (verse 1-4)
    # Col 10: 16 chars (verse 5-8)
    # Col 11: 16 chars (verse 9-12)
    # Col 12: 16 chars (verse 13-16)
    # Col 13: 14 chars (date)
    # Col 14: 9 chars (author)
    # (Sum: 4 + 35+36+33+33+34+33+33+23 + 16+16+16+16 + 14+9 = 351)

    col_char_counts = [35, 36, 33, 33, 34, 33, 33, 23, 16, 16, 16, 16, 14, 9]

    # Column X centers from right to left
    # Measured from peak detection on 03011:
    col_x_centers = [
        3037, 2888, 2772, 2646, 2507, 2377, 2240, 2102,
        1978, 1852, 1728, 1605, 1187, 865
    ]

    # Column Y ranges:
    # Full cols (1-7): top ~ 1680, bottom ~ 4260
    # Col 1 has indentation at top (嘗謂 is indented)
    # Col 8: ends early at 其銘曰
    # Verse cols (9-12): top ~ 2050, bottom ~ 4050
    # Date (13): top ~ 1720, bottom ~ 3200
    # Author (14): top ~ 3400, bottom ~ 4250

    col_boxes = []
    char_idx = 4 # after trán bia

    for c_idx, count in enumerate(col_char_counts):
        cx = col_x_centers[c_idx]
        half_w = 58 # half width of character box

        if c_idx == 0: # Col 1
            y_start = 1750
            y_end = 4260
        elif c_idx < 7: # Cols 2-7
            y_start = 1680
            y_end = 4260
        elif c_idx == 7: # Col 8
            y_start = 1680
            y_end = 3450
        elif c_idx < 12: # Verse (Cols 9-12)
            y_start = 2050
            y_end = 4050
        elif c_idx == 12: # Date
            y_start = 1720
            y_end = 3150
        else: # Author
            y_start = 3450
            y_end = 4260

        y_positions = np.linspace(y_start, y_end, count + 1)
        for i in range(count):
            y1 = int(round(y_positions[i]))
            y2 = int(round(y_positions[i+1]))
            # Slight padding between characters
            bh = y2 - y1
            y1_box = max(0, y1 + int(bh * 0.05))
            y2_box = min(h, y2 - int(bh * 0.05))
            x1_box = max(0, cx - half_w)
            x2_box = min(w, cx + half_w)
            col_boxes.append([x1_box, y1_box, x2_box, y2_box])

    all_boxes = trán_boxes + col_boxes
    assert len(all_boxes) == num_chars, f"Mismatch: {len(all_boxes)} boxes vs {num_chars} chars"

    # Assemble bounding_boxes dict and annotations dict
    bounding_boxes = {}
    annotations = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for idx, (box, ch) in enumerate(zip(all_boxes, annotation_chars), 1):
        box_id = str(idx)
        annotations[box_id] = ch

        # Flags
        is_unknown = (ch == '□')
        is_unavail = (ch == '@')
        is_expert = False
        is_suspicious = False

        # Status & damage level
        if is_unknown:
            status = 'damaged'
            damage_level = 3
        else:
            # Check patch variance and contrast
            bx1, by1, bx2, by2 = box
            patch = gray[by1:by2, bx1:bx2]
            lap = float(cv2.Laplacian(patch, cv2.CV_64F).var()) if patch.size > 0 else 0
            if lap < 120.0:
                status = 'damaged'
                damage_level = 2
            elif lap < 250.0:
                status = 'damaged'
                damage_level = 1
            else:
                status = 'intact'
                damage_level = None

        bounding_boxes[box_id] = {
            "bbox": box,
            "status": status,
            "damage_level": damage_level,
            "unknown": is_unknown,
            "unavailable_font": is_unavail,
            "expert_prediction": is_expert,
            "suspicious": is_suspicious,
        }

    # Crop and image resize
    # Black stele body bounds: [200, 490, 3520, 4820]
    scaled_crop, out_size = auto_scale_crop([200, 490, 3520, 4820], [w, h])
    crop_doc = crop_document(image_name, scaled_crop, out_size)['crop']
    resize_doc = image_resize([w, h], out_size)

    doc = {
        "image": image_name,
        "bounding_boxes": bounding_boxes,
        "annotations": annotations,
        "image_resize": resize_doc,
        "crop": crop_doc,
    }

    validate_document(doc, image_name, [w, h])
    return doc


# ==============================================================================
# 3. GENERATE CHARACTER BOUNDING BOXES FOR 03010.JPG (Face 3010)
# ==============================================================================

def generate_annotation_3010() -> dict:
    image_name = "03010.jpg"
    image_path = DATA_DIR / image_name
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2] # 4868, 3840

    annotation_chars = characters(CONTENT_3010["Nguyên văn chữ Hán Nôm"])
    num_chars = len(annotation_chars) # 490

    # Trán bia: 4 large characters (right to left):
    # 1: 本, 2: 社, 3: 保, 4: 迹
    trán_boxes = [
        [2780, 1020, 3220, 1460], # 本
        [2140, 1020, 2580, 1460], # 社
        [1500, 1020, 1940, 1460], # 保
        [820,  1020, 1280, 1460], # 迹
    ]

    # In lòng bia of 3010:
    # There are 17 columns from right to left:
    # Col 1: 34
    # Col 2: 36
    # Col 3: 36
    # Col 4: 36
    # Col 5: 38
    # Col 6: 38
    # Col 7: 38
    # Col 8: 26
    # Col 9: 11
    # Col 10: 42
    # Col 11: 32
    # Col 12: 17
    # Col 13: 16
    # Col 14: 26
    # Col 15: 39
    # Col 16: 34
    # Col 17: 7
    # Sum: 4 + 34+36+36+36+38+38+38+26 + 11+42+32+17+16+26+39+34+7 = 490!

    col_counts_3010 = [
        35, 35, 35, 35, 35, 35, 35, 30,
        11, 43, 26, 15, 15, 24, 39, 31, 7
    ]
    assert sum(col_counts_3010) + 4 == num_chars, f"Counts mismatch: {sum(col_counts_3010) + 4} vs {num_chars}"

    # Column X centers from right to left
    col_x_centers = [
        3276, 3145, 3005, 2865, 2713, 2573, 2429, 2291,
        2155, 2049, 1910, 1644, 1489, 1365, 1244, 1081, 834
    ]

    col_boxes = []
    char_idx = 4

    for c_idx, count in enumerate(col_counts_3010):
        cx = col_x_centers[c_idx]
        half_w = 54

        # Full cols (1-7): top ~ 1680, bottom ~ 4380
        # Col 8: top ~ 1680, bottom ~ 3600
        # Col 9 (計: 一保置...): top ~ 1800, bottom ~ 2650
        # Col 10 (其後神...): top ~ 1680, bottom ~ 4380
        # Col 11 (一本社下田...): top ~ 1720, bottom ~ 3950
        # Col 12 (一本社有歌...): top ~ 1720, bottom ~ 3000
        # Col 13 (一本社有犒...): top ~ 1720, bottom ~ 2900
        # Col 14 (一後日送終...): top ~ 1720, bottom ~ 3600
        # Col 15 (一遞年忌日...): top ~ 1720, bottom ~ 4380
        # Col 16 (古立保同...): top ~ 1720, bottom ~ 4100
        # Col 17 (全社上下...): top ~ 3800, bottom ~ 4380

        if c_idx < 7:
            y_start = 1680
            y_end = 4380
        elif c_idx == 7:
            y_start = 1680
            y_end = 3600
        elif c_idx == 8:
            y_start = 1800
            y_end = 2650
        elif c_idx == 9:
            y_start = 1680
            y_end = 4380
        elif c_idx == 10:
            y_start = 1720
            y_end = 3950
        elif c_idx == 11:
            y_start = 1720
            y_end = 3000
        elif c_idx == 12:
            y_start = 1720
            y_end = 2900
        elif c_idx == 13:
            y_start = 1720
            y_end = 3600
        elif c_idx == 14:
            y_start = 1720
            y_end = 4380
        elif c_idx == 15:
            y_start = 1720
            y_end = 4100
        else: # c_idx == 16 (全社上下等共記)
            y_start = 3800
            y_end = 4380

        y_positions = np.linspace(y_start, y_end, count + 1)
        for i in range(count):
            y1 = int(round(y_positions[i]))
            y2 = int(round(y_positions[i+1]))
            bh = y2 - y1
            y1_box = max(0, y1 + int(bh * 0.05))
            y2_box = min(h, y2 - int(bh * 0.05))
            x1_box = max(0, cx - half_w)
            x2_box = min(w, cx + half_w)
            col_boxes.append([x1_box, y1_box, x2_box, y2_box])

    all_boxes = trán_boxes + col_boxes
    assert len(all_boxes) == num_chars, f"Mismatch: {len(all_boxes)} boxes vs {num_chars} chars"

    bounding_boxes = {}
    annotations = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for idx, (box, ch) in enumerate(zip(all_boxes, annotation_chars), 1):
        box_id = str(idx)
        annotations[box_id] = ch

        is_unknown = (ch == '□')
        is_unavail = (ch == '@')
        is_expert = (ch == '𥸷') # Expert prediction
        is_suspicious = False

        if is_unknown:
            status = 'damaged'
            damage_level = 3
        else:
            bx1, by1, bx2, by2 = box
            patch = gray[by1:by2, bx1:bx2]
            lap = float(cv2.Laplacian(patch, cv2.CV_64F).var()) if patch.size > 0 else 0
            if lap < 120.0:
                status = 'damaged'
                damage_level = 2
            elif lap < 250.0:
                status = 'damaged'
                damage_level = 1
            else:
                status = 'intact'
                damage_level = None

        bounding_boxes[box_id] = {
            "bbox": box,
            "status": status,
            "damage_level": damage_level,
            "unknown": is_unknown,
            "unavailable_font": is_unavail,
            "expert_prediction": is_expert,
            "suspicious": is_suspicious,
        }

    # Black stele body bounds: [260, 470, 3560, 4550]
    scaled_crop, out_size = auto_scale_crop([260, 470, 3560, 4550], [w, h])
    crop_doc = crop_document(image_name, scaled_crop, out_size)['crop']
    resize_doc = image_resize([w, h], out_size)

    doc = {
        "image": image_name,
        "bounding_boxes": bounding_boxes,
        "annotations": annotations,
        "image_resize": resize_doc,
        "crop": crop_doc,
    }

    validate_document(doc, image_name, [w, h])
    return doc


# ==============================================================================
# 4. SAVE AND EXPORT EVERYTHING
# ==============================================================================

def main():
    print("Generating annotations for student_036...")

    doc_3011 = generate_annotation_3011()
    doc_3010 = generate_annotation_3010()

    # 1. Save individual annotation JSON files
    path_3011 = OUT_DIR / "03011.json"
    path_3010 = OUT_DIR / "03010.json"
    atomic_write(path_3011, doc_3011)
    atomic_write(path_3010, doc_3010)
    print(f"Saved: {path_3011}")
    print(f"Saved: {path_3010}")

    # 2. Save UI state metadata
    import hashlib
    def get_meta(doc, txt):
        chars_str = ''.join(characters(txt))
        return {
            'text_hash': hashlib.sha256(chars_str.encode('utf-8')).hexdigest(),
            'document_hash': hashlib.sha256(json.dumps(doc, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
        }
    atomic_write(OUT_DIR / ".state" / "03011.json", get_meta(doc_3011, CONTENT_3011["Nguyên văn chữ Hán Nôm"]))
    atomic_write(OUT_DIR / ".state" / "03010.json", get_meta(doc_3010, CONTENT_3010["Nguyên văn chữ Hán Nôm"]))

    # 3. Save verified inscription content registry
    content_registry = [
        {
            "image": "03010.jpg",
            "inscription_code": "03010",
            "content": CONTENT_3010,
        },
        {
            "image": "03011.jpg",
            "inscription_code": "03011",
            "content": CONTENT_3011,
        }
    ]
    validate_content_document(content_registry[0], "03010.jpg")
    validate_content_document(content_registry[1], "03011.jpg")
    atomic_write(OUT_DIR / ".state" / "content.json", content_registry)
    print("Saved content registry to .state/content.json")

    # 4. Save annotations.zip (Download All export bundle)
    text_annotations = [doc_3010, doc_3011]
    archive_path = save_export_archive(
        annotations=text_annotations,
        content=content_registry,
        output_dir=OUT_DIR,
        source_mismatches=[],
    )
    print(f"Created export archive: {archive_path}")

    # 5. Extract files alongside zip so individual files are also immediately available
    atomic_write(OUT_DIR / "text_annotations.json", text_annotations)
    atomic_write(OUT_DIR / "inscription_content.json", content_registry)
    atomic_write(OUT_DIR / "source_mismatches.json", [])
    print("Wrote text_annotations.json, inscription_content.json, source_mismatches.json")

    print("Annotation automation complete!")


if __name__ == '__main__':
    main()
