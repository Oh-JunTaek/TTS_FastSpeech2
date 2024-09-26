""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''


# Regular expression matching whitespace:
import re
from unidecode import unidecode
from .numbers import normalize_numbers
from pykospacing import Spacing
from hanspell import spell_checker
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text

# 한국어 숫자 변환 함수 (예: '일' -> '1')
def normalize_korean_numbers(text):
    num_dict = {
        '일': '1', '이': '2', '삼': '3', '사': '4', '오': '5',
        '육': '6', '칠': '7', '팔': '8', '구': '9', '영': '0'
    }
    for key, value in num_dict.items():
        text = text.replace(key, value)
    return text

def expand_korean_abbreviations(text):
    """한국어 약어 확장 (예: 'ㄱㄱ' -> '고고')"""
    abbreviations = {
        'ㄱㄱ': '고고',
        'ㄴㄴ': '노노',
        # 필요에 따라 약어 목록 추가
    }
    for abbr, expansion in abbreviations.items():
        text = text.replace(abbr, expansion)
    return text

def remove_special_characters(text):
    """특수 문자 제거 (한글, 영어, 숫자, 공백 제외)"""
    return re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)

def correct_spacing(text):
    """띄어쓰기 교정"""
    spacing = Spacing()
    return spacing(text)

def correct_spelling(text):
    """맞춤법 교정"""
    spelled_sent = spell_checker.check(text)
    return spelled_sent.checked

def collapse_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def korean_cleaners(text):
    """한국어 텍스트 클리너"""
    print(f"Original text: {text}")
    
    # # 1. 맞춤법 교정 (일단 비활성화)
    # text = correct_spelling(text)
    
    # 2. 숫자 변환 (한글 -> 숫자)
    text = normalize_korean_numbers(text)
    print(f"After number normalization: {text}")
    
    # 3. 한국어 약어 확장
    text = expand_korean_abbreviations(text)
    print(f"After abbreviation expansion: {text}")
    
    # 4. 특수 문자 제거
    text = remove_special_characters(text)
    print(f"After special character removal: {text}")
    
    # 5. 띄어쓰기 교정
    text = correct_spacing(text)
    print(f"After spacing correction: {text}")
    
    # 6. 공백 정리
    text = collapse_whitespace(text)
    print(f"Final cleaned text: {text}")
    
    return text