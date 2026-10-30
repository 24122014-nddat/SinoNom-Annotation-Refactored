import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from annotation.workflow import Workflow, _find_persisted_in_zip
from annotation.state import new_state

class DummyOptions:
    content_titles = ('Nguyên văn chữ Hán Nôm', 'Phiên âm Hán Việt', 'Toát yếu')
    metadata_fields = (('Tên bia', 'ten_bia'), ('Địa điểm', 'dia_diem'), ('Niên đại', 'nien_dai'))
    annotation_title = 'Nguyên văn chữ Hán Nôm'
    output_dir = 'student_036/annotations'
    source_json = 'student_036/student_036.json'
    skip_detection = False

class ProcessedDataLoadTests(unittest.TestCase):
    def setUp(self):
        self.options = DummyOptions()
        self.engine = Workflow(self.options)

    def test_find_persisted_in_zip_1446(self):
        p = Path('processed_data/01446.jpg')
        if not p.exists():
            self.skipTest('processed_data/01446.jpg not found')
        res = _find_persisted_in_zip(p, self.options.output_dir)
        self.assertIsNotNone(res)
        self.assertTrue(res['is_mismatch'])
        self.assertEqual(len(res['doc']['bounding_boxes']), 315)
        self.assertIn('［', res['content']['Phiên âm Hán Việt'])

    def test_open_image_processed_1446(self):
        p = Path('processed_data/01446.jpg')
        if not p.exists():
            self.skipTest('processed_data/01446.jpg not found')
        state = self.engine.open_image(p)
        self.assertEqual(state['code'], '01446')
        self.assertTrue(state['detection_loaded'])
        self.assertEqual(len(state['regions']), 315)
        self.assertIsNotNone(state.get('source_mismatch'))
        self.assertEqual(state['source_mismatch']['issue_type'], 'missing_text')
        self.assertTrue(state['workflow']['alignment_valid'])
        self.assertIsNotNone(state.get('crop'))
        # Ensure brackets were restored in Phiên âm Hán Việt
        sino = state['draft_content']['noi_dung'][0]['chuyen_muc'][1]['van_ban']
        self.assertIn('［］', sino)

    def test_open_image_processed_10622(self):
        p = Path('processed_data/10622.jpg')
        if not p.exists():
            self.skipTest('processed_data/10622.jpg not found')
        state = self.engine.open_image(p)
        self.assertEqual(state['code'], '10622')
        self.assertTrue(state['detection_loaded'])
        self.assertEqual(len(state['regions']), 103)
        self.assertIsNone(state.get('source_mismatch'))

    def test_open_image_processed_3010(self):
        p = Path('processed_data/03010.jpg')
        if not p.exists():
            self.skipTest('processed_data/03010.jpg not found')
        state = self.engine.open_image(p)
        self.assertEqual(state['code'], '03010')
        self.assertTrue(state['detection_loaded'])
        # In student_036/annotations, 03010.json has 490 boxes
        self.assertEqual(len(state['regions']), 490)
        self.assertTrue(state['workflow']['alignment_valid'])
        sino = state['draft_content']['noi_dung'][0]['chuyen_muc'][1]['van_ban']
        self.assertIn('［］', sino)
        self.assertIn('Nguyễn Ðăng Sĩ', sino)

    def test_open_image_fallback_to_zip_when_no_output(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            options = DummyOptions()
            options.output_dir = tmpdir
            engine = Workflow(options)
            p = Path('processed_data/03010.jpg')
            if not p.exists():
                self.skipTest('processed_data/03010.jpg not found')
            state = engine.open_image(p)
            self.assertEqual(state['code'], '03010')
            self.assertTrue(state['detection_loaded'])
            # From zip file directly: 497 boxes
            self.assertEqual(len(state['regions']), 497)

if __name__ == '__main__':
    unittest.main()
