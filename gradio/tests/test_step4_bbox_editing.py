"""Unit tests for Step 4 (Status & Order) bounding box editing, sorting, and status preservation."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from app import create_app, parser
from annotation.state import new_state, set_verified_content, initialize_alignment
from annotation.bbox import add_bbox
from annotation.status import update_status, confirm_status
from annotation.io import atomic_write
from annotation.workflow import Workflow


class DummyOptions:
    content_titles = ('Nguyên văn chữ Hán Nôm', 'Phiên âm Hán Việt', 'Toát yếu')
    metadata_fields = (('Tên bia', 'ten_bia'), ('Địa điểm', 'dia_diem'), ('Niên đại', 'nien_dai'))
    annotation_title = 'Nguyên văn chữ Hán Nôm'
    output_dir = 'student_036/annotations'
    source_json = 'student_036/student_036.json'
    skip_detection = True


class Step4BBoxEditingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.options = DummyOptions()
        self.options.output_dir = self.tmp
        self.workflow = Workflow(self.options)

    def make_step4_state(self, chars="永寺樂", num_boxes=3):
        state = new_state()
        state.update(
            image='1.png', image_size=[500, 500], current_step=4, code='1',
            draft_content={'noi_dung': [{'ky_hieu': '1', 'chuyen_muc': [
                {'tieu_de': 'Nguyên văn chữ Hán Nôm', 'van_ban': chars}]}]}
        )
        set_verified_content(state, {}, chars)
        for i in range(num_boxes):
            x = 100 * (i + 1)
            add_bbox(state, [x, 50, x + 80, 150])
        initialize_alignment(state)
        return state

    def test_step4_actions_accepted(self):
        """Workflow actions: add, update, delete, sort_boxes, confirm_source_mismatch work in Step 4."""
        s = self.make_step4_state()
        self.assertEqual(s['current_step'], 4)

        # 1. Update bbox in Step 4
        uid = s['region_uid_by_box_id']['1']
        s = self.workflow.apply(s, 'update', {'uid': uid, 'bbox': [110, 50, 190, 150]})
        self.assertEqual(s['regions'][uid]['bbox'], [110, 50, 190, 150])

        # 2. Add bbox in Step 4
        s = self.workflow.apply(s, 'add', {'bbox': [400, 50, 480, 150]})
        self.assertEqual(len(s['regions']), 4)

        # 3. Confirm mismatch in Step 4 (4 boxes > 3 characters = missing_text)
        s = self.workflow.apply(s, 'confirm_source_mismatch', {'issue_type': 'missing_text', 'note': ''})
        self.assertIsNotNone(s['source_mismatch'])
        self.assertEqual(s['source_mismatch']['issue_type'], 'missing_text')

        # 4. Sort boxes in Step 4
        s = self.workflow.apply(s, 'sort_boxes')
        self.assertEqual(len(s['bounding_boxes']), 4)
        self.assertEqual(len(s['reading_order']), 4)

        # 5. Delete bbox in Step 4
        del_uid = s['region_uid_by_box_id']['4']
        s = self.workflow.apply(s, 'delete', {'ids': [del_uid]})
        self.assertEqual(len(s['regions']), 3)

    def test_status_preserved_by_identity_when_sorting_step4(self):
        """Physical box status (damaged level 2, unknown) follows the box geometry/identity, not the index."""
        s = new_state()
        chars = "甲乙"
        s.update(
            image='1.png', image_size=[500, 500], current_step=4, code='1',
            draft_content=[{'noi_dung': [{'ky_hieu': '1', 'chuyen_muc': [
                {'tieu_de': 'Nguyên văn chữ Hán Nôm', 'van_ban': chars}]}]}]
        )
        set_verified_content(s, {}, chars)

        # Box A at x=200..280
        uid_a = add_bbox(s, [200, 50, 280, 150])
        # Box B at x=300..380
        uid_b = add_bbox(s, [300, 50, 380, 150])
        initialize_alignment(s)

        # In Sino-Nom reading order (R-to-L), x=300 is Box 1, x=200 is Box 2
        id_b_initial = s['box_id_by_region'][uid_b]
        id_a_initial = s['box_id_by_region'][uid_a]

        # Mark Box B as damaged level 2, unknown=True
        update_status(s, uid_b, 'damaged', unknown=True, damage_level=2)
        confirm_status(s)
        self.assertEqual(s['bounding_boxes'][id_b_initial]['status'], 'damaged')
        self.assertEqual(s['bounding_boxes'][id_b_initial]['damage_level'], 2)
        self.assertTrue(s['bounding_boxes'][id_b_initial]['unknown'])

        # Now in Step 4, we add Box C at x=400..480 (further to the right, becomes the new Box 1)
        uid_c = add_bbox(s, [400, 50, 480, 150])
        self.assertEqual(s['regions'][uid_c]['status'], 'intact')

        # 3 boxes vs 2 characters -> confirm missing_text
        s = self.workflow.apply(s, 'confirm_source_mismatch', {'issue_type': 'missing_text', 'note': ''})

        # Sort boxes:
        # Expected spatial order (R-to-L): C (x=400) -> B (x=300) -> A (x=200)
        # So Box 1 = C, Box 2 = B, Box 3 = A!
        s = self.workflow.apply(s, 'sort_boxes')

        self.assertEqual(s['region_uid_by_box_id']['1'], uid_c)
        self.assertEqual(s['region_uid_by_box_id']['2'], uid_b)
        self.assertEqual(s['region_uid_by_box_id']['3'], uid_a)

        # Verify Box 2 (which is Box B) STILL has damaged level 2, unknown=True!
        self.assertEqual(s['bounding_boxes']['2']['status'], 'damaged')
        self.assertEqual(s['bounding_boxes']['2']['damage_level'], 2)
        self.assertTrue(s['bounding_boxes']['2']['unknown'])

        # Verify Box 1 (Box C) is intact
        self.assertEqual(s['bounding_boxes']['1']['status'], 'intact')
        self.assertIsNone(s['bounding_boxes']['1']['damage_level'])

        # Verify Box 3 (Box A) is intact
        self.assertEqual(s['bounding_boxes']['3']['status'], 'intact')

    def test_step4_ui_components_visible_in_render(self):
        """In Step 4, render() returns box_group, mismatch_group, and status_group visible."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / '1.png'
            Image.new('RGB', (100, 100)).save(image)
            source = root / 'source.json'
            doc = [{'noi_dung': [{'ky_hieu': '1', 'chuyen_muc': [
                {'tieu_de': 'Nguyên văn chữ Hán Nôm', 'van_ban': '永寺'}]}]}]
            atomic_write(source, doc)
            app = create_app(parser().parse_args([
                '--image-dir', str(root), '--source-json', str(source),
                '--output-dir', str(root / 'out'), '--skip-detection']))

            # Find render function via an action callback
            state = self.make_step4_state('永寺', num_boxes=2)
            state.update(image_path=str(image), image_url='image.jpg', draft_content=doc[0])
            ctx = dict(active=state, drafts={})

            apply_changes = next(f.fn for f in app.fns.values()
                                 if f.fn and f.fn.__name__ == 'apply_status_changes')
            payload = {'statuses': {'1': 'intact', '2': 'intact'},
                       'damage_levels': {'1': None, '2': None},
                       'unknowns': {'1': False, '2': False},
                       'textSequence': ['永', '寺']}
            result = apply_changes(ctx, json.dumps(payload), 'intact')

            # outputs indices:
            # 9: box_group
            # 15: status_group
            # 31: mismatch_group
            # 36: box_color_group
            self.assertTrue(result[9].get('visible', False))
            self.assertTrue(result[15].get('visible', False))
            self.assertTrue(result[31].get('visible', False))
            self.assertTrue(result[36].get('visible', False))

    def test_step4_advance_to_step6_after_editing_and_mismatch(self):
        """User edits boxes in Step 4, confirms mismatch, sorts, and advances to Step 6."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / '1.png'
            Image.new('RGB', (100, 100)).save(image)
            source = root / 'source.json'
            atomic_write(source, [{'noi_dung': [{'ky_hieu': '1', 'chuyen_muc': [
                {'tieu_de': 'Nguyên văn chữ Hán Nôm', 'van_ban': '永寺'}]}]}])
            options = parser().parse_args([
                '--image-dir', str(root), '--source-json', str(source),
                '--output-dir', str(root / 'out'), '--skip-detection'])
            app = create_app(options)
            engine = Workflow(options)

            open_image = next(f.fn for f in app.fns.values() if f.fn and f.fn.__name__ == 'open_image')
            advance_stream = next(f.fn for f in app.fns.values() if f.fn and f.fn.__name__ == 'next_with_progress')
            def advance(*args, **kwargs):
                return list(advance_stream(*args, **kwargs))[-1]

            opened = open_image(dict(active=new_state(), drafts={}), str(image))
            ctx = opened[0]
            draft = opened[44]
            # Advance step 1 -> step 2
            ctx = advance(ctx)[0]
            self.assertEqual(ctx['active']['current_step'], 2)
            # Advance step 2 -> step 3
            ctx = advance(ctx, content_draft=draft)[0]
            self.assertEqual(ctx['active']['current_step'], 3)

            # In Step 3: create 2 boxes and sort
            add_bbox(ctx['active'], [10, 10, 40, 40])
            add_bbox(ctx['active'], [50, 10, 80, 40])
            from annotation.bbox import sync_draft_boxes
            orders = {u: i for i, u in enumerate(ctx['active']['regions'], 1)}
            boxes = {u: b['bbox'] for u, b in ctx['active']['regions'].items()}
            sync_draft_boxes(ctx['active'], {'boxes': boxes, 'orders': orders})

            # Advance step 3 -> step 4
            ctx = advance(ctx)[0]
            self.assertEqual(ctx['active']['current_step'], 4)

            # In Step 4: Delete 1 box (now 1 box vs 2 characters = extra_text)
            del_uid = list(ctx['active']['regions'].keys())[0]
            from annotation.bbox import delete_bbox
            delete_bbox(ctx['active'], del_uid)
            self.assertEqual(len(ctx['active']['regions']), 1)

            # Confirm source mismatch extra_text
            ctx['active'] = engine.apply(
                ctx['active'], 'confirm_source_mismatch', {'issue_type': 'extra_text', 'note': ''})

            # Sort boxes in Step 4
            ctx['active'] = engine.apply(ctx['active'], 'sort_boxes')

            # Advance step 4 -> step 6
            advanced = advance(ctx, issue_type='extra_text')
            self.assertEqual(advanced[0]['active']['current_step'], 6)


if __name__ == '__main__':
    unittest.main()
