import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from annotation.state import new_state
from annotation.bbox import add_bbox, delete_bbox, sync_draft_boxes


class BoxOrderReset(unittest.TestCase):
    def make_state(self):
        state = new_state()
        state['image_size'] = [100, 100]
        state['regions'] = {
            'a': dict(bbox=[0, 0, 10, 10], status='intact', order=1),
            'b': dict(bbox=[20, 0, 30, 10], status='intact', order=2),
        }
        return state

    def test_add_and_delete_clear_every_order(self):
        for action in ('add', 'delete'):
            with self.subTest(action=action):
                state = self.make_state()
                if action == 'add':
                    add_bbox(state, [40, 0, 50, 10])
                else:
                    delete_bbox(state, 'a')
                self.assertTrue(all(box['order'] is None for box in state['regions'].values()))
                self.assertEqual(state['reading_order'], [])

    def test_commit_membership_change_ignores_stale_orders(self):
        state = self.make_state()
        boxes = dict(state['regions'])
        boxes['c'] = dict(bbox=[40, 0, 50, 10], status='intact', order=3)
        sync_draft_boxes(state, dict(boxes=boxes, orders={'a': 1, 'b': 2, 'c': 3}))
        self.assertTrue(all(box['order'] is None for box in state['regions'].values()))

    def test_coordinate_edit_keeps_orders(self):
        state = self.make_state()
        boxes = {uid: dict(box) for uid, box in state['regions'].items()}
        boxes['a']['bbox'] = [1, 1, 11, 11]
        sync_draft_boxes(state, dict(boxes=boxes), materialize_alignment=False)
        self.assertEqual([box['order'] for box in state['regions'].values()], [1, 2])
