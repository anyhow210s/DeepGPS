from __future__ import annotations

import html
import json
import shutil
import zipfile
from pathlib import Path
from PIL import Image

ROOT = Path('/Users/koscom/Documents/DeepGPS')
OUT = ROOT / 'outputs' / 'photo_geolocation_presentation_korean.pptx'
WORK = ROOT / 'outputs' / '019e8b33-2adc-7321-ba88-1d6f3f779776' / 'presentations' / 'photo-geolocation-korean'
IMG_TRAIN = ROOT / 'outputs' / 'figures' / 'training_history.png'
IMG_CM = ROOT / 'outputs' / 'figures' / 'confusion_matrix.png'

SLIDE_W = 12192000
SLIDE_H = 6858000
EMU = 9525  # 1280 x 720 canvas maps to PowerPoint widescreen

COLORS = {
    'paper': 'F7F3EA',
    'ink': '17212B',
    'muted': '5E6975',
    'blue': '2F6BFF',
    'blue2': 'DDE7FF',
    'amber': 'F4A340',
    'amber2': 'FFF0DA',
    'green': '2E8B57',
    'green2': 'DFF2E7',
    'red': 'D95D5D',
    'red2': 'F9E1E1',
    'line': 'D9D1C4',
    'white': 'FFFFFF',
}

FONT = 'Apple SD Gothic Neo'


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def emu(v: float) -> int:
    return int(round(v * EMU))


def solid_fill(color: str) -> str:
    return f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'


def line(color='D9D1C4', width=1) -> str:
    return f'<a:ln w="{int(width*12700)}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:ln>'


def no_line() -> str:
    return '<a:ln><a:noFill/></a:ln>'


def shape_xml(idx, x, y, w, h, fill=None, outline=None, radius=False):
    geom = 'roundRect' if radius else 'rect'
    fill_xml = solid_fill(fill) if fill else '<a:noFill/>'
    line_xml = no_line() if outline is None else line(outline)
    return f'''
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{idx}" name="Shape {idx}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
      <p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>{fill_xml}{line_xml}</p:spPr>
      <p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>
    </p:sp>'''


def text_paragraph(text, size=24, color='17212B', bold=False, align='l'):
    b = ' b="1"' if bold else ''
    return f'''<a:p><a:pPr algn="{align}"/><a:r><a:rPr lang="ko-KR" sz="{int(size*100)}"{b}><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="{FONT}"/><a:ea typeface="{FONT}"/></a:rPr><a:t>{esc(text)}</a:t></a:r></a:p>'''


def textbox_xml(idx, x, y, w, h, paragraphs, fill=None, outline=None, radius=False, margin=10):
    geom = 'roundRect' if radius else 'rect'
    fill_xml = solid_fill(fill) if fill else '<a:noFill/>'
    line_xml = no_line() if outline is None else line(outline)
    body = ''.join(paragraphs)
    return f'''
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{idx}" name="TextBox {idx}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>{fill_xml}{line_xml}</p:spPr>
      <p:txBody><a:bodyPr lIns="{emu(margin)}" tIns="{emu(margin)}" rIns="{emu(margin)}" bIns="{emu(margin)}" wrap="square"/><a:lstStyle/>{body}</p:txBody>
    </p:sp>'''


def image_xml(idx, rel_id, x, y, w, h, name):
    return f'''
    <p:pic>
      <p:nvPicPr><p:cNvPr id="{idx}" name="{esc(name)}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
      <p:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
      <p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
    </p:pic>'''


def footer(slide_no):
    return textbox_xml(900 + slide_no, 70, 682, 1140, 24, [text_paragraph(f'Photo Geolocation Prediction using Deep Learning   |   {slide_no}/10', 9, '8A8177')], margin=0)


def title(kicker, claim):
    return (
        textbox_xml(10, 70, 42, 220, 28, [text_paragraph(kicker, 12, '2F6BFF', True)], margin=0)
        + textbox_xml(11, 70, 78, 820, 92, [text_paragraph(claim, 30, '17212B', True)], margin=0)
    )


def metric_card(idx, x, y, w, h, value, label, context, color='blue'):
    c = COLORS[color]
    bg = COLORS.get(color + '2', 'DDE7FF')
    return textbox_xml(idx, x, y, w, h, [
        text_paragraph(value, 30, c, True, 'ctr'),
        text_paragraph(label, 13, '17212B', True, 'ctr'),
        text_paragraph(context, 10, '5E6975', False, 'ctr'),
    ], fill=bg, outline=None, radius=True, margin=12)


def bullet_box(idx, x, y, w, h, heading, bullets, color='blue'):
    paras = [text_paragraph(heading, 18, COLORS[color], True)]
    for b in bullets:
        paras.append(text_paragraph('• ' + b, 14, '17212B'))
    return textbox_xml(idx, x, y, w, h, paras, fill='FFFFFF', outline='E4DDD1', radius=True, margin=18)


def slide_xml(shapes, bg='F7F3EA'):
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:bg><p:bgPr>{solid_fill(bg)}<a:effectLst/></p:bgPr></p:bg><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    {shapes}
  </p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''


def slide_rels(image_targets, slide_no):
    rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>']
    for i, target in enumerate(image_targets, start=2):
        rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{target}"/>')
    rels.append(f'<Relationship Id="rId99" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide{slide_no}.xml"/>')
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + ''.join(rels) + '</Relationships>'


def note_para(text, size=1200, bold=False):
    b = ' b="1"' if bold else ''
    return f'<a:p><a:r><a:rPr lang="ko-KR" sz="{size}"{b}><a:solidFill><a:srgbClr val="17212B"/></a:solidFill><a:latin typeface="{FONT}"/><a:ea typeface="{FONT}"/></a:rPr><a:t>{esc(text)}</a:t></a:r></a:p>'


def notes_slide_xml(slide_no, notes):
    body = ''.join(note_para(line, 1400 if i == 0 else 1100, bold=(i == 0)) for i, line in enumerate(notes))
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:notes xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    <p:sp>
      <p:nvSpPr><p:cNvPr id="2" name="Speaker Notes {slide_no}"/><p:cNvSpPr txBox="1"/><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr>
      <p:spPr><a:xfrm><a:off x="685800" y="685800"/><a:ext cx="5486400" cy="7772400"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
      <p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>{body}</p:txBody>
    </p:sp>
  </p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:notes>'''


def notes_rels(slide_no):
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="../slides/slide{slide_no}.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster" Target="../notesMasters/notesMaster1.xml"/>
</Relationships>'''


def notes_master_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:notesMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
</p:notesMaster>'''


def notes_master_rels():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>'''


def speaker_notes():
    return [
        [
            '발표 노트: 프로젝트 한 문장 소개',
            '이 프로젝트는 사진 안의 시각적 단서만 보고 위치 카테고리를 예측하는 딥러닝 이미지 분류 프로젝트입니다.',
            '중요한 점은 GPS 좌표나 EXIF 위치 정보를 쓰지 않는다는 것입니다. 즉, 사진 파일에 숨어 있는 위치 정보가 아니라 이미지 자체를 보고 판단합니다.',
            'ResNet50이라는 CNN 기반 사전학습 모델을 사용했고, 전체 과정은 로컬에서 실행됩니다.',
            '현재 학습 결과는 17개 위치 클래스, 전체 427장 이미지, 테스트 정확도 74.07%입니다.',
        ],
        [
            '발표 노트: 문제 정의',
            '입력은 사진 한 장이고, 출력은 이 사진이 어느 위치 클래스에 가까운지에 대한 확률입니다.',
            '예를 들어 모델은 San Francisco 같은 정확한 GPS 좌표를 맞히는 것이 아니라, 학습된 후보 위치 중 어디에 가장 가까운지를 분류합니다.',
            '이 문제가 어려운 이유는 장소가 서로 비슷하게 보일 수 있고, 같은 장소라도 날씨, 시간, 각도에 따라 사진이 달라지기 때문입니다.',
            '그래서 이 프로젝트의 목표는 완벽한 지도 위치 예측이 아니라, 데이터 준비부터 학습, 평가, 데모까지 이어지는 딥러닝 파이프라인을 완성하는 것입니다.',
        ],
        [
            '발표 노트: 데이터셋 설명',
            '데이터는 사용자가 미리 위치별 폴더로 수동 분류한 로컬 이미지입니다.',
            '각 폴더 이름이 하나의 클래스가 됩니다. 예를 들어 Yosemite라는 폴더가 있으면 Yosemite가 예측 가능한 클래스가 됩니다.',
            'prepare_dataset.py는 이미지 파일만 읽고 JSON, 동영상, 손상된 이미지는 제외합니다.',
            '전체 데이터는 70% train, 15% validation, 15% test로 나눴습니다. 이번 학습에서는 train 290장, validation 56장, test 81장이 사용됐습니다.',
            '주의할 점은 클래스 불균형입니다. 어떤 위치는 사진이 많고 어떤 위치는 매우 적어서, 성능 해석에 영향을 줍니다.',
        ],
        [
            '발표 노트: 전체 파이프라인',
            '파이프라인은 data/raw에서 시작합니다. 위치별 폴더에 들어 있는 이미지를 읽고, data/processed에 train/val/test 구조로 다시 저장합니다.',
            '그 다음 ResNet50 모델을 학습하고, validation accuracy가 가장 좋은 모델을 best_resnet50.pth로 저장합니다.',
            '평가 단계에서는 test set으로 accuracy, precision, recall, F1-score, confusion matrix를 계산합니다.',
            '마지막으로 Streamlit UI에서 새 사진을 업로드하고 top-3 예측과 Grad-CAM을 확인합니다.',
            '이 모든 과정은 외부 AI API가 아니라 로컬 PyTorch 코드로 실행됩니다.',
        ],
        [
            '발표 노트: ResNet50 모델 설명',
            'ResNet50은 CNN, 즉 Convolutional Neural Network 기반 이미지 분류 모델입니다.',
            'CNN은 이미지의 작은 패턴부터 시작해서 선, 색, 질감, 모양, 장면 구조 같은 특징을 단계적으로 추출합니다.',
            'ResNet의 핵심 아이디어는 residual connection입니다. 층이 깊어져도 학습이 잘 되도록 입력 정보를 일부 건너뛰어 전달하는 구조입니다.',
            '이번 프로젝트에서는 ImageNet으로 사전학습된 ResNet50을 가져왔습니다. ImageNet 사전학습이란 이미 대규모 이미지 데이터로 일반적인 시각 특징을 배운 상태라는 뜻입니다.',
            '기존 ResNet50의 마지막 분류층은 ImageNet 1000개 클래스를 위한 것이므로, 이 부분을 우리 데이터의 17개 위치 클래스로 바꿨습니다.',
            '이 방식을 전이학습이라고 합니다. 이미 배운 시각 특징은 활용하고, 마지막 분류 부분만 새로운 문제에 맞게 다시 학습하는 방식입니다.',
        ],
        [
            '발표 노트: 학습 설정 설명',
            '학습은 두 단계로 진행했습니다.',
            '첫 번째 단계는 classifier head 학습입니다. ResNet50의 CNN backbone은 고정하고, 마지막 fully connected layer만 학습했습니다. 이렇게 하면 작은 데이터셋에서 과적합을 줄일 수 있습니다.',
            '두 번째 단계는 fine-tuning입니다. 마지막 ResNet block인 layer4를 열어서 우리 위치 데이터에 조금 더 맞게 조정했습니다.',
            'Loss는 CrossEntropyLoss를 사용했습니다. 다중 클래스 분류에서 예측 확률이 정답 클래스와 얼마나 다른지 측정하는 손실 함수입니다.',
            'Optimizer는 Adam을 사용했습니다. Adam은 학습률을 적응적으로 조절해서 딥러닝에서 자주 쓰이는 최적화 방법입니다.',
            'Best model은 validation accuracy가 가장 높은 epoch의 모델입니다. 이번 결과에서는 epoch 8이 best checkpoint였습니다.',
        ],
        [
            '발표 노트: 결과 수치 해석',
            'Validation accuracy 82.14%는 학습 중 validation set에서 가장 좋았던 정확도입니다. 이 값을 기준으로 best model을 저장했습니다.',
            'Test accuracy 74.07%는 학습과 모델 선택에 사용하지 않은 test set에서의 최종 성능입니다. 따라서 실제 일반화 성능에 더 가까운 숫자입니다.',
            'Macro F1 0.68은 각 클래스를 같은 비중으로 평균낸 F1-score입니다. 클래스별 데이터 수가 불균형할 때 특히 중요합니다.',
            'Weighted F1 0.71은 클래스별 샘플 수를 반영한 평균입니다. 사진이 많은 클래스가 더 큰 영향을 줍니다.',
            '그래프에서 loss가 줄고 accuracy가 올라가는 흐름은 모델이 학습되고 있다는 신호입니다. 다만 validation보다 test가 낮기 때문에 일반화에는 아직 한계가 있습니다.',
        ],
        [
            '발표 노트: 평가 지표와 혼동 행렬',
            'Accuracy는 전체 테스트 이미지 중 맞춘 비율입니다. 직관적이지만 클래스 불균형이 있으면 충분하지 않을 수 있습니다.',
            'Precision은 모델이 어떤 클래스로 예측한 것 중 실제로 맞은 비율입니다. 예를 들어 Yosemite라고 예측한 사진 중 진짜 Yosemite가 얼마나 되는지를 봅니다.',
            'Recall은 실제 해당 클래스 사진 중 모델이 찾아낸 비율입니다. 예를 들어 실제 Yosemite 사진 중 모델이 Yosemite라고 맞힌 비율입니다.',
            'F1-score는 precision과 recall의 균형을 보는 지표입니다. 둘 중 하나만 높아서는 좋은 모델이라고 보기 어렵기 때문에 F1을 함께 봅니다.',
            'Confusion matrix는 어떤 클래스가 어떤 클래스로 헷갈렸는지 보여주는 표입니다. 대각선 값이 많을수록 잘 맞힌 것이고, 대각선 밖의 값은 오분류입니다.',
            '이번 결과에서는 일부 클래스의 F1이 낮습니다. 특히 테스트 이미지 수가 적거나 시각적으로 비슷한 클래스는 성능이 불안정할 수 있습니다.',
        ],
        [
            '발표 노트: 시연 방법',
            '시연은 Streamlit UI에서 진행하면 됩니다. 먼저 Data & Training 탭은 데이터 준비와 학습 실행을 보여주는 탭입니다.',
            '이미 학습이 끝났다면 Training Results 탭에서 학습 곡선, classification report, confusion matrix를 보여주면 됩니다.',
            'Photo Prediction 탭에서는 새 사진을 업로드하고 top-3 예측 결과를 보여줍니다. top-3는 가장 가능성이 높은 위치 후보 3개와 확률입니다.',
            'Grad-CAM은 모델이 이미지의 어느 부분을 보고 판단했는지 보여주는 설명 가능성 기법입니다.',
            '발표할 때는 “이 heatmap이 밝은 부분일수록 모델 판단에 더 영향을 준 영역입니다”라고 설명하면 됩니다.',
        ],
        [
            '발표 노트: 한계와 개선 방향',
            '이번 프로젝트의 장점은 작은 데이터셋으로도 end-to-end 딥러닝 시스템을 완성했다는 점입니다.',
            '하지만 한계도 명확합니다. 클래스별 사진 수가 균형적이지 않고, 어떤 클래스는 테스트 이미지가 2~5장 정도로 매우 적습니다.',
            '그래서 특정 클래스의 precision, recall, F1-score는 데이터 수에 따라 크게 흔들릴 수 있습니다.',
            '개선 방향으로는 클래스별 사진 수를 균형 있게 늘리고, 비슷한 장소를 더 잘 구분할 수 있도록 다양한 각도와 조명의 사진을 추가하는 것이 있습니다.',
            '또한 EfficientNetB0나 Vision Transformer와 비교해서 ResNet50이 가장 좋은 선택인지 실험해볼 수 있습니다.',
            '마지막으로 unknown threshold를 추가하면, 학습된 위치 중 어느 곳에도 확신이 없을 때 “모름”이라고 답하게 만들 수 있습니다.',
        ],
    ]


def make_slides():
    slides = []
    # 1 cover
    s = ''
    s += shape_xml(2, 0, 0, 1280, 720, fill='17212B')
    s += textbox_xml(3, 70, 70, 780, 160, [text_paragraph('Photo Geolocation\nPrediction', 42, 'FFFFFF', True), text_paragraph('using Deep Learning', 26, 'F4A340', True)], margin=0)
    s += textbox_xml(4, 72, 250, 720, 90, [text_paragraph('EXIF GPS 없이 사진 자체의 시각적 특징으로 위치 카테고리를 예측하는 ResNet50 전이학습 프로젝트', 19, 'F7F3EA')], margin=0)
    for i, (val, lab, ctx, col) in enumerate([('17', '위치 클래스', '수동 분류된 로컬 사진', 'blue'), ('427', '전체 이미지', 'train/val/test 분할', 'amber'), ('74.07%', 'Test Accuracy', '학습에 쓰지 않은 테스트셋', 'green')], start=5):
        s += metric_card(i, 70 + (i-5)*250, 450, 220, 120, val, lab, ctx, col)
    s += textbox_xml(20, 900, 520, 300, 70, [text_paragraph('Demo: Streamlit UI에서 학습 결과와 예측을 시연', 15, 'F7F3EA')], margin=0)
    slides.append((slide_xml(s, '17212B'), []))

    # 2 problem
    s = title('PROBLEM', 'GPS 메타데이터 없이 사진만 보고 위치를 맞춘다')
    s += bullet_box(20, 70, 210, 370, 260, '문제 정의', ['입력: JPG/PNG/HEIC 사진 한 장', '출력: 위치 클래스별 확률', 'GPS/EXIF 정보는 사용하지 않음'], 'blue')
    s += bullet_box(21, 470, 210, 350, 260, '왜 어려운가?', ['장소 간 시각적 유사성', '개인 사진의 데이터 불균형', '날씨/조명/각도 차이'], 'amber')
    s += bullet_box(22, 850, 210, 350, 260, '프로젝트 목표', ['작은 데이터셋으로 완성형 파이프라인 구현', '학습, 평가, 예측, 데모까지 연결', '설명 가능한 Grad-CAM 포함'], 'green')
    s += footer(2)
    slides.append((slide_xml(s), []))

    # 3 dataset
    s = title('DATASET', '위치별로 수동 분류한 로컬 사진 폴더를 학습 데이터로 만들었다')
    s += metric_card(20, 70, 190, 190, 105, '290', 'Train', '모델 학습용', 'blue')
    s += metric_card(21, 280, 190, 190, 105, '56', 'Validation', 'best model 선택', 'amber')
    s += metric_card(22, 490, 190, 190, 105, '81', 'Test', '최종 평가용', 'green')
    s += metric_card(23, 700, 190, 190, 105, '17', 'Classes', '위치 카테고리', 'red')
    s += bullet_box(24, 70, 340, 520, 210, '전처리', ['JSON sidecar와 동영상은 제외', '손상된 이미지는 skip', '이미지는 224x224로 변환', 'ImageNet mean/std로 정규화'], 'blue')
    s += bullet_box(25, 630, 340, 520, 210, '중요한 관찰', ['Golfland USA는 train 110장으로 가장 많음', '일부 클래스는 test 2~5장뿐임', '결과 해석 시 클래스 불균형을 함께 봐야 함'], 'red')
    s += footer(3)
    slides.append((slide_xml(s), []))

    # 4 pipeline
    s = title('PIPELINE', '데이터 준비부터 데모까지 하나의 로컬 딥러닝 시스템으로 연결했다')
    steps = [('Raw\nphotos', 'data/raw'), ('Prepare\ndataset', '70/15/15 split'), ('ResNet50', 'transfer learning'), ('Evaluate', 'accuracy + F1'), ('Demo UI', 'Streamlit')]
    for i, (a, b) in enumerate(steps):
        x = 80 + i*235
        s += textbox_xml(30+i, x, 265, 170, 110, [text_paragraph(a, 23, '17212B', True, 'ctr'), text_paragraph(b, 12, '5E6975', False, 'ctr')], fill='FFFFFF', outline='D9D1C4', radius=True, margin=10)
        if i < len(steps)-1:
            s += textbox_xml(50+i, x+178, 300, 50, 28, [text_paragraph('→', 26, '2F6BFF', True, 'ctr')], margin=0)
    s += bullet_box(70, 150, 470, 980, 95, '핵심 원칙', ['온라인 사진 서비스나 외부 AI API를 사용하지 않고, 로컬 폴더와 로컬 PyTorch 학습만 사용한다.'], 'green')
    s += footer(4)
    slides.append((slide_xml(s), []))

    # 5 model
    s = title('MODEL', 'ImageNet으로 사전학습된 ResNet50을 위치 분류기로 바꿨다')
    layers = [('Input\n224x224', '사진'), ('ResNet50\nBackbone', '시각 특징 추출'), ('New FC\nLayer', '17개 위치 클래스'), ('Softmax', '확률 출력')]
    for i, (a, b) in enumerate(layers):
        x = 105 + i*270
        color = ['blue','blue','amber','green'][i]
        s += textbox_xml(30+i, x, 250, 210, 120, [text_paragraph(a, 24, COLORS[color], True, 'ctr'), text_paragraph(b, 12, '5E6975', False, 'ctr')], fill='FFFFFF', outline='D9D1C4', radius=True, margin=12)
        if i < 3:
            s += textbox_xml(50+i, x+220, 290, 45, 30, [text_paragraph('→', 26, '17212B', True, 'ctr')], margin=0)
    s += bullet_box(80, 120, 445, 1040, 105, '발표 설명 문장', ['ResNet50은 CNN 기반 이미지 분류 모델이며, 기존에 학습한 일반 이미지 특징을 활용하기 때문에 작은 개인 사진 데이터셋에서도 학습을 시작하기 좋습니다.'], 'blue')
    s += footer(5)
    slides.append((slide_xml(s), []))

    # 6 training settings
    s = title('TRAINING', '처음에는 분류기만, 이후에는 마지막 ResNet 블록까지 미세조정했다')
    s += textbox_xml(20, 80, 205, 500, 130, [text_paragraph('1단계: Classifier Head', 24, '2F6BFF', True), text_paragraph('Epoch 1–5', 16, '17212B', True), text_paragraph('Backbone은 고정하고 마지막 분류층만 학습', 14, '5E6975')], fill='DDE7FF', outline=None, radius=True, margin=18)
    s += textbox_xml(21, 700, 205, 500, 130, [text_paragraph('2단계: Fine-tuning', 24, 'F4A340', True), text_paragraph('Epoch 6–8', 16, '17212B', True), text_paragraph('ResNet layer4를 열어 위치 데이터에 더 맞게 조정', 14, '5E6975')], fill='FFF0DA', outline=None, radius=True, margin=18)
    s += bullet_box(22, 80, 395, 500, 160, '주요 설정', ['Loss: CrossEntropyLoss', 'Optimizer: Adam', 'Batch size: UI 기본 16', 'Best model 기준: validation accuracy'], 'blue')
    s += bullet_box(23, 700, 395, 500, 160, '왜 이렇게 했나?', ['처음부터 전체 모델을 학습하면 과적합 위험이 큼', '분류기 먼저 적응 후 일부 CNN block만 조정', '작은 데이터셋에서 안정적인 전이학습 전략'], 'green')
    s += footer(6)
    slides.append((slide_xml(s), []))

    # 7 results with training image
    s = title('RESULTS', '검증 정확도는 82.14%, 테스트 정확도는 74.07%로 마무리됐다')
    s += metric_card(20, 70, 180, 190, 105, '82.14%', 'Best Val Acc', 'epoch 8 checkpoint', 'blue')
    s += metric_card(21, 280, 180, 190, 105, '74.07%', 'Test Acc', '81 test images', 'green')
    s += metric_card(22, 490, 180, 190, 105, '0.68', 'Macro F1', '클래스 동일 가중', 'amber')
    s += metric_card(23, 700, 180, 190, 105, '0.71', 'Weighted F1', '샘플 수 반영', 'red')
    s += image_xml(40, 'rId2', 105, 330, 500, 250, 'training_history.png')
    s += bullet_box(41, 650, 335, 500, 235, '해석', ['학습이 진행되며 loss는 감소하고 accuracy는 상승', 'fine-tuning 이후 validation accuracy가 더 개선', 'test accuracy가 validation보다 낮아 일반화 성능은 보수적으로 봐야 함'], 'green')
    s += footer(7)
    slides.append((slide_xml(s), ['image1.png']))

    # 8 metrics and confusion matrix
    s = title('METRICS', 'Accuracy만 보지 않고 precision, recall, F1과 혼동 행렬을 함께 봤다')
    s += image_xml(20, 'rId2', 70, 190, 520, 395, 'confusion_matrix.png')
    s += bullet_box(21, 640, 185, 520, 95, 'Accuracy', ['전체 테스트 중 맞춘 비율. 현재 74.07%.'], 'blue')
    s += bullet_box(22, 640, 295, 520, 95, 'Precision', ['이 위치라고 예측한 것 중 실제로 맞은 비율.'], 'amber')
    s += bullet_box(23, 640, 405, 520, 95, 'Recall', ['실제 해당 위치 사진 중 모델이 찾아낸 비율.'], 'green')
    s += bullet_box(24, 640, 515, 520, 95, 'F1-score', ['Precision과 Recall의 균형. 데이터가 불균형할 때 중요.'], 'red')
    s += footer(8)
    slides.append((slide_xml(s), ['image2.png']))

    # 9 demo
    s = title('DEMO', 'Streamlit UI에서 학습 실행과 예측 결과 확인까지 시연한다')
    demos = [('1', 'Data & Training', '사진 폴더 경로 입력\nPrepare and Train 실행'), ('2', 'Training Results', '학습 곡선\nclassification report\nconfusion matrix 확인'), ('3', 'Photo Prediction', '새 사진 업로드\nTop-3 예측 확률 확인'), ('4', 'Grad-CAM', '모델이 본 영역을\nheatmap으로 설명')]
    for i, (num, head, body) in enumerate(demos):
        x = 75 + i*292
        s += textbox_xml(20+i, x, 245, 245, 220, [text_paragraph(num, 30, '2F6BFF', True, 'ctr'), text_paragraph(head, 18, '17212B', True, 'ctr'), text_paragraph(body, 13, '5E6975', False, 'ctr')], fill='FFFFFF', outline='D9D1C4', radius=True, margin=14)
    s += textbox_xml(50, 115, 515, 1050, 55, [text_paragraph('시연 멘트: “이 UI는 외부 API가 아니라 로컬 모델 파일 best_resnet50.pth를 불러와 예측합니다.”', 16, '17212B', True, 'ctr')], fill='DFF2E7', outline=None, radius=True, margin=10)
    s += footer(9)
    slides.append((slide_xml(s), []))

    # 10 lessons
    s = title('LESSONS', '작동하는 파이프라인은 만들었지만, 데이터 품질이 성능을 크게 좌우했다')
    s += bullet_box(20, 70, 205, 350, 260, '잘 된 점', ['ResNet50 전이학습으로 빠르게 baseline 구현', '학습/평가/예측/Grad-CAM/UI까지 연결', '실제 테스트셋 기준 74.07% 달성'], 'green')
    s += bullet_box(21, 465, 205, 350, 260, '한계', ['클래스별 이미지 수가 불균형', '일부 클래스 test support가 너무 작음', '비슷한 장소는 혼동 가능'], 'red')
    s += bullet_box(22, 860, 205, 350, 260, '다음 단계', ['클래스별 사진 수 균형 맞추기', 'EfficientNetB0 / ViT 비교', 'unknown location threshold 추가', 'EXIF는 모델 입력이 아닌 정리 보조로만 사용'], 'blue')
    s += textbox_xml(60, 120, 540, 1040, 55, [text_paragraph('결론: 작은 데이터셋에서도 전이학습을 사용하면 완성형 딥러닝 데모를 만들 수 있지만, 실제 성능은 데이터 균형과 테스트셋 크기에 크게 의존한다.', 15, '17212B', True, 'ctr')], fill='FFF0DA', outline=None, radius=True, margin=10)
    s += footer(10)
    slides.append((slide_xml(s), []))
    return slides


def content_types(slide_count):
    overrides = [
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Default Extension="png" ContentType="image/png"/>',
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/notesMasters/notesMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesMaster+xml"/>',
    ]
    for i in range(1, slide_count+1):
        overrides.append(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
        overrides.append(f'<Override PartName="/ppt/notesSlides/notesSlide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/>')
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' + ''.join(overrides) + '</Types>'


def presentation_xml(slide_count):
    sld_ids = ''.join(f'<p:sldId id="{255+i}" r:id="rId{i}"/>' for i in range(1, slide_count+1))
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{slide_count+1}"/></p:sldMasterIdLst><p:sldIdLst>{sld_ids}</p:sldIdLst><p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>'''


def presentation_rels(slide_count):
    rels = ''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>' for i in range(1, slide_count+1))
    rels += f'<Relationship Id="rId{slide_count+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    rels += f'<Relationship Id="rId{slide_count+2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>'
    rels += f'<Relationship Id="rId{slide_count+3}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster" Target="notesMasters/notesMaster1.xml"/>'
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + rels + '</Relationships>'


def minimal_theme():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="DeepGPS"><a:themeElements><a:clrScheme name="DeepGPS"><a:dk1><a:srgbClr val="17212B"/></a:dk1><a:lt1><a:srgbClr val="F7F3EA"/></a:lt1><a:dk2><a:srgbClr val="5E6975"/></a:dk2><a:lt2><a:srgbClr val="FFFFFF"/></a:lt2><a:accent1><a:srgbClr val="2F6BFF"/></a:accent1><a:accent2><a:srgbClr val="F4A340"/></a:accent2><a:accent3><a:srgbClr val="2E8B57"/></a:accent3><a:accent4><a:srgbClr val="D95D5D"/></a:accent4><a:accent5><a:srgbClr val="8A8177"/></a:accent5><a:accent6><a:srgbClr val="D9D1C4"/></a:accent6><a:hlink><a:srgbClr val="2F6BFF"/></a:hlink><a:folHlink><a:srgbClr val="6B4EFF"/></a:folHlink></a:clrScheme><a:fontScheme name="DeepGPS"><a:majorFont><a:latin typeface="Apple SD Gothic Neo"/><a:ea typeface="Apple SD Gothic Neo"/><a:cs typeface="Arial"/></a:majorFont><a:minorFont><a:latin typeface="Apple SD Gothic Neo"/><a:ea typeface="Apple SD Gothic Neo"/><a:cs typeface="Arial"/></a:minorFont></a:fontScheme><a:fmtScheme name="DeepGPS"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>'''


def minimal_master():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst></p:sldMaster>'''


def minimal_layout():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'''


def write_pptx():
    slides = make_slides()
    notes = speaker_notes()
    if len(notes) != len(slides):
        raise ValueError(f"Expected {len(slides)} notes entries, got {len(notes)}")
    media = {'image1.png': IMG_TRAIN, 'image2.png': IMG_CM}
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', content_types(len(slides)))
        z.writestr('_rels/.rels', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>')
        z.writestr('ppt/presentation.xml', presentation_xml(len(slides)))
        z.writestr('ppt/_rels/presentation.xml.rels', presentation_rels(len(slides)))
        z.writestr('ppt/theme/theme1.xml', minimal_theme())
        z.writestr('ppt/slideMasters/slideMaster1.xml', minimal_master())
        z.writestr('ppt/slideMasters/_rels/slideMaster1.xml.rels', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>')
        z.writestr('ppt/slideLayouts/slideLayout1.xml', minimal_layout())
        z.writestr('ppt/slideLayouts/_rels/slideLayout1.xml.rels', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>')
        z.writestr('ppt/notesMasters/notesMaster1.xml', notes_master_xml())
        z.writestr('ppt/notesMasters/_rels/notesMaster1.xml.rels', notes_master_rels())
        for i, (xml, imgs) in enumerate(slides, start=1):
            z.writestr(f'ppt/slides/slide{i}.xml', xml)
            z.writestr(f'ppt/slides/_rels/slide{i}.xml.rels', slide_rels(imgs, i))
            z.writestr(f'ppt/notesSlides/notesSlide{i}.xml', notes_slide_xml(i, notes[i - 1]))
            z.writestr(f'ppt/notesSlides/_rels/notesSlide{i}.xml.rels', notes_rels(i))
        for target, source in media.items():
            z.write(source, f'ppt/media/{target}')
    print(OUT)

if __name__ == '__main__':
    write_pptx()
