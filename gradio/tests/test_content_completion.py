"""Configured content defaults and annotation-only persistence."""
import json
import sys
import tempfile
import unittest
import zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from annotation.export import collect_content_documents, save_export_archive
from annotation.io import atomic_write, read_json
from annotation.state import new_state
from annotation.text_extraction import (
    complete_content_record, content_fields, edit_content_field,
    save_content_document,
)
from annotation.workflow import Workflow

TITLES = ('Nguyên văn chữ Hán Nôm', 'Phiên âm Hán Việt', 'Toát yếu')
METADATA = (('Tên bia', 'ten_bia'), ('Địa điểm', 'dia_diem'))
LABELS = tuple(label for label, _ in METADATA) + TITLES


class ContentCompletion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.image = self.root / '001.png'
        Image.new('RGB', (100, 100)).save(self.image)
        self.record = {'extra': 'keep', 'noi_dung': [
            {'ky_hieu': '2', 'chuyen_muc': []},
            {'ky_hieu': '1', 'chuyen_muc': [
                {'tieu_de': 'Mục khác', 'van_ban': 'giữ nguyên'},
                {'tieu_de': TITLES[0], 'van_ban': '永寺'},
            ]},
        ]}
        self.source = self.root / 'source.json'
        atomic_write(self.source, [self.record])
        self.source_bytes = self.source.read_bytes()
        self.output = self.root / 'out'
        self.engine = Workflow(SimpleNamespace(
            source_json=self.source, output_dir=self.output,
            content_titles=TITLES, metadata_fields=METADATA,
            annotation_title=TITLES[0], skip_detection=True))
        self.addCleanup(self.engine._preview_cache.cleanup)

    def fields(self, record):
        return content_fields(record, '001', TITLES, METADATA)

    def edit(self, state, title, value):
        field = next(field for field in self.fields(state['draft_content'])
                     if field['title'] == title)
        return self.engine.apply(state, 'field', {'path': field['path'], 'value': value})

    def review(self, state):
        state = self.engine.apply(state, 'next')
        for x in (0, 20):
            state = self.engine.apply(state, 'add', {'bbox': [x, 0, x + 10, 10]})
        state = self.engine.apply(state, 'sort_boxes')
        state = self.engine.apply(state, 'next')
        state = self.engine.apply(state, 'reorder_text', {'sequence': ['寺', '永']})
        while state['current_step'] < 7:
            state = self.engine.apply(state, 'next')
        return state

    def test_defaults_are_ordered_editable_and_do_not_mutate_source(self):
        fields = self.fields(self.record)
        self.assertEqual([field['title'] for field in fields], list(LABELS))
        self.assertEqual([field['value'] for field in fields], ['', '', '永寺', '', ''])
        prepared = complete_content_record(self.record, '001', TITLES, METADATA)
        self.assertEqual(self.fields(prepared), fields)
        self.assertEqual(complete_content_record(prepared, '001', TITLES, METADATA), prepared)
        for field in fields:
            updated = edit_content_field(self.record, '001', field['path'], 'bổ sung', TITLES, METADATA)
            self.assertEqual(next(f for f in self.fields(updated)
                                  if f['title'] == field['title'])['value'], 'bổ sung')
            self.assertEqual(updated['noi_dung'][0], self.record['noi_dung'][0])
        self.assertNotIn('ten_bia', self.record)
        self.assertEqual(len(self.record['noi_dung'][1]['chuyen_muc']), 2)

    def test_unconfigured_and_other_face_paths_are_rejected(self):
        for path in [('extra',), ('noi_dung', 0, 'chuyen_muc', 0, 'van_ban'),
                     ('noi_dung', 1, 'chuyen_muc', 0, 'van_ban')]:
            with self.assertRaises(ValueError):
                edit_content_field(self.record, '001', path, 'changed', TITLES, METADATA)

    def test_save_reopen_export_and_clear_supplement(self):
        state = self.engine.open_image(self.image)
        state = self.edit(state, 'Tên bia', 'Tên bia 漢\nDòng hai')
        state = self.edit(state, 'Toát yếu', 'Nội dung bổ sung\n𠀀')
        state = self.review(state)
        self.assertFalse((self.output / '.state/content.json').exists())
        state = self.engine.apply(state, 'save')
        saved = read_json(self.output / '.state/content.json')[0]
        self.assertEqual(list(saved['content']), list(LABELS))
        documents = collect_content_documents([self.image], self.output, titles=LABELS)
        archive = save_export_archive([], documents, self.output)
        with zipfile.ZipFile(archive) as bundle:
            self.assertEqual(json.loads(bundle.read('inscription_content.json')), [saved])
        reopened = self.engine.open_image(self.image)
        self.assertEqual(reopened['annotations'], state['annotations'])
        self.assertEqual(reopened['reading_order'], state['reading_order'])
        values = {field['title']: field['value'] for field in self.fields(reopened['draft_content'])}
        self.assertEqual(values, saved['content'])
        reopened = self.edit(reopened, 'Toát yếu', '')
        reopened = self.engine.apply(reopened, 'next')
        while reopened['current_step'] < 7:
            reopened = self.engine.apply(reopened, 'next')
        reopened = self.engine.apply(reopened, 'save')
        for key in ('bounding_boxes', 'annotations', 'reading_order'):
            self.assertEqual(reopened[key], state[key])
        values = {f['title']: f['value'] for f in self.fields(
            self.engine.open_image(self.image)['draft_content'])}
        self.assertEqual(values['Toát yếu'], '')
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_supplement_keeps_manual_alignment_but_source_edit_invalidates(self):
        state = self.engine.apply(self.review(self.engine.open_image(self.image)), 'save')
        while state['current_step'] > 2:
            state = self.engine.apply(state, 'back')
        keys = ('bounding_boxes', 'annotations', 'reading_order', 'regions',
                'text_sequence', 'text_token_ids', 'box_id_by_region', 'region_uid_by_box_id')
        before = {key: deepcopy(state[key]) for key in keys}
        state = self.edit(state, 'Toát yếu', 'Bổ sung')
        state = self.engine.apply(state, 'save_content')
        self.assertEqual({key: state[key] for key in keys}, before)
        self.assertTrue(state['workflow']['alignment_valid'])
        state = self.edit(state, TITLES[0], '永寺文')
        state = self.engine.apply(state, 'save_content')
        self.assertFalse(state['workflow']['alignment_valid'])
        self.assertEqual(state['annotations'], {})

    def test_legacy_nulls_and_missing_keys_restore_and_export_as_empty(self):
        self.engine.apply(self.review(self.engine.open_image(self.image)), 'save')
        legacy = {'image': self.image.name, 'inscription_code': '001',
                  'content': {TITLES[0]: '永寺', 'Tên bia': None}}
        atomic_write(self.output / '.state/content.json', [legacy])
        reopened = self.engine.open_image(self.image)
        self.assertEqual([f['value'] for f in self.fields(reopened['draft_content'])],
                         ['', '', '永寺', '', ''])
        documents = collect_content_documents([self.image], self.output, titles=LABELS)
        self.assertEqual(list(documents[0]['content']), list(LABELS))
        self.assertTrue(all(isinstance(value, str) for value in documents[0]['content'].values()))
        save_content_document(documents[0], self.output, LABELS)
        self.assertEqual(read_json(self.output / '.state/content.json'), documents)

    def test_source_annotation_section_is_still_required(self):
        source = deepcopy(self.record)
        source['noi_dung'][1]['chuyen_muc'] = []
        atomic_write(self.source, [source])
        with self.assertRaisesRegex(ValueError, 'section with text content is required'):
            self.engine.open_image(self.image)

    def test_ui_exposes_and_submits_missing_fields(self):
        from app import create_app, parser
        config = self.root / 'config.json'
        atomic_write(config, {'records': {
            'metadata': [{'label': label, 'field': field, 'type': 'string'}
                         for label, field in METADATA],
            'content': {'start_heading': TITLES[0], 'section_headings': list(TITLES)}}})
        app = create_app(parser().parse_args([
            '--config', str(config), '--image-dir', str(self.root),
            '--source-json', str(self.source), '--output-dir', str(self.output),
            '--skip-detection']))
        callbacks = {f.fn.__name__: f.fn for f in app.fns.values() if f.fn}
        opened = callbacks['open_image']({'active': new_state(), 'drafts': {}}, str(self.image))
        open_callback = next(f for f in app.fns.values() if f.fn and f.fn.__name__ == 'open_image')
        draft_index = next(i for i, component in enumerate(open_callback.outputs)
                           if component.elem_id == 'content-draft-bridge')
        draft = json.loads(opened[draft_index])
        self.assertEqual([field['title'] for field in draft.values()], list(LABELS))
        for field in draft.values():
            if field['title'] == 'Toát yếu':
                self.assertEqual(field['value'], '')
                field['value'] = 'Bổ sung\n漢'
        result = list(callbacks['next_with_progress'](
            opened[0], json.dumps(draft, ensure_ascii=False)))[-1]
        self.assertEqual(result[0]['active']['current_step'], 3)
        self.assertEqual(next(f for f in self.fields(result[0]['active']['draft_content'])
                              if f['title'] == 'Toát yếu')['value'], 'Bổ sung\n漢')
        self.assertEqual(self.source.read_bytes(), self.source_bytes)


if __name__ == '__main__':
    unittest.main()
