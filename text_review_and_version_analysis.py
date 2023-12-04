import os
import io
import re
from fuzzywuzzy import fuzz
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import streamlit as st

# 共同代碼
def split_sentences(content):
    # 按標點切分句子
    pattern = r'[。！？；)）\n]'
    sentences = re.split(pattern, content)
    return [s.strip() for s in sentences if s.strip()]

def is_punctuation(char):
    # 檢查字符是否是標點符號
    return char in "、，,。.！!？?；;:：（）()[]【】“”《》·"

def find_punctuation_indices(sentence, is_punctuation):
    # 找到標點符號所在的索引
    indices = [i for i, char in enumerate(sentence) if is_punctuation(char)]
    return indices

def remove_punctuation(text):
    # 刪除標點符號。
    # 首先檢查是否是字符串，如果不是就返回原始值，避免因空值導致在df中apply失敗。
    if isinstance(text, str):
        return re.sub(r'[、，,。.！!？?；;:：（）()\[\]【】“”‘’《》·]', '', text)
    else:
        return text

def is_similar(input_sentence,sentence):
    '''
    input_sentence:被對比的句子；
    sentence：用來對比的句子。
    return：返回布爾值，代表兩個句子是否相似。
    '''
    is_similar = False
    similarity = fuzz.partial_ratio(input_sentence,sentence) # 進行非完全匹配。避免有些句子衹是由於標點時多用了逗號導致句子過長，而無法與相似的短句匹配。
    input_sentence_len = len(remove_punctuation(input_sentence))
    sentence_len = len(remove_punctuation(sentence))
    if similarity > 60:  # 相似度閾值
        if input_sentence_len > 5 and sentence_len > 5: #  匹配字符較多的才可以算作相似，否則衹是偶然相同。特別是書名，例如“《湖南通志》”加上書名號正好6個字符。
            is_similar = True
        elif input_sentence_len > 5 and sentence_len < 6:
            if sentence_len > input_sentence_len / 2:
                is_similar = True #如果用來對比的句子雖短，但不短於原句的一半，也算相似
        else:
            # 由於剩下的都是input_sentence_len < 6的句子，如果字符過少，可以忽略。例如可能是小節標題，“歲時”“冠婚”之類的
            if input_sentence_len  > 2 and sentence_len  > 2:
                is_similar = True
    return is_similar

    
def similar_char_index(input_sentence, sentence):
    '''
    功能：找到sentence中與input_sentence不同的內容,返回其在sentenc中的索引
    '''
    # 首先通過長度爲2的滑動窗口獲取相同內容的索引
    input_sentence = remove_punctuation(input_sentence) # 刪除標點符號，避免受其干擾
    keywords = [input_sentence[i:i+2] for i in range(len(input_sentence)-1)] # 滑動窗口獲取輸入的兩兩字符組合
    indices = set()
    for keyword in keywords:
        matches = re.finditer(re.escape(keyword), sentence)
        for match in matches:
            indices.add(match.start())
            indices.add(match.start() + 1)
    
    # 由於滑動窗口找到的字符會漏掉句子或子句開頭或結尾相似的字符，因此要補充查找這一部分。
    # 句子或子句的開頭：句子開頭、標點符號後第一個字符；
    # 句子或子句的結尾：句子結尾、標點符號前第一個字符。
    punctuation_indices =  find_punctuation_indices(sentence, is_punctuation) # 找到標點符號的索引
    # 開頭索引
    segment_start_indices = [0]
    segment_start_indices += list(map(lambda x: x + 1, punctuation_indices))
    # 結尾索引
    sentenc_len = len(sentence)
    end = sentenc_len - 1
    segment_end_indices = [end] # 不直接用[-1]以免出現重複索引導致渲染時多出字符
    segment_end_indices += list(map(lambda x: x - 1, punctuation_indices))
    # 找到需要用來匹配的字符
    start_dic = {}
    for i in segment_start_indices:
        if i + 2 < sentenc_len: # 索引上限
            pattern = sentence[i] + '.'
            pattern = pattern + sentence[i+2]
            start_dic[i] = pattern

    end_dic = {}
    for i in segment_end_indices:
        if i - 2 > -1: # 索引下限
            pattern = sentence[i-2] + '.'
            pattern = pattern + sentence[i]
            end_dic[i] = pattern

    all_pattern = {**start_dic, **end_dic}
    for k,v in all_pattern.items():
        pattern = re.compile(v)
        match = re.search(pattern, input_sentence)
        if match:
            indices.add(k)
    return indices

# 文本對勘

def highlight_diff(sentence, indices):
    # 根據indices索引列表渲染sentence，將索引內的字符和標點符號渲染成藍色，其他的字符渲染成紅色
    # 也就是根據相似內容索引，將相似的內容渲染成藍色，不同的內容渲染成紅色
    similar_sentence = ''.join([f"<font color='blue'>{sentence[i]}</font>" 
                                if i in indices or is_punctuation(sentence[i]) 
                                else sentence[i] for i in range(len(sentence))])
    # 渲染其他字符
    highlighted_sentence = "<font color='red'>" + similar_sentence + "</font>"
    return highlighted_sentence


def text_review():
    st.subheader("文本對勘")
    # 上傳多個txt文件
    uploaded_files = st.file_uploader("請上傳utf-8編碼的txt文件作爲對校本:", type=["txt"], accept_multiple_files=True)
    
    if uploaded_files:
        # 讀取txt文件內容
        txt_contents = {uploaded_file.name: uploaded_file.read().decode('utf-8').lstrip('\ufeff') for uploaded_file in uploaded_files}
    # 文本輸入框
    text = st.text_area("請輸入需對勘的文本（按Ctrl+Enter執行）：")
    
    if text:
        # 按標點切分句子
        input_sentences = split_sentences(text)
        
        # 遍歷每個輸入句子，與上傳的txt文件中的每個句子比較
        for input_sentence in input_sentences:
            similar_sentences = []
            for txt_file, txt_content in txt_contents.items():
                for txt_sentence in split_sentences(txt_content):
                    condition = is_similar(input_sentence,txt_sentence)
                    if condition:
                        similar_sentences.append((txt_file, txt_sentence))    
            
            if similar_sentences:
                output_str = f"{input_sentence} <font color='green'>【"
                for i, (file, sentence) in enumerate(similar_sentences):
                    file_name = os.path.splitext(file)[0]  # 去除後綴
                    char_index = similar_char_index(input_sentence, sentence)
                    highlight_diff_sentence = highlight_diff(sentence, char_index)
                    output_str += f"{file_name}：{highlight_diff_sentence}"
                    if i < len(similar_sentences) - 1:
                        output_str += "；"
                output_str += "】</font>"
                st.markdown(output_str, unsafe_allow_html=True)
            else:
                st.write(input_sentence)


# 版本分析
def split_sentences_plus():
    st.write('此處版本分析主要針對地方志設計，用來分析其內容的來源，其他古籍若有此需求亦適用。')
    st.write('具體分析方法是輸入所有可能來源的txt文件，並且文件名中包含出版年。如果該文件出版年最早，則不分析其來源。其他文件則找到與該文本內容相似的其他書籍內容，且衹關注出版年在其之前、年代最早的相似句。結果會導出具體的最早相似句表格及其來源的統計餅圖。')
    # 添加文件上傳器
    uploaded_files = st.file_uploader("請上傳多個utf-8編碼的txt文件:", type="txt", accept_multiple_files=True)
    # 檢查上傳的文件
    if not uploaded_files:
        st.warning("請上傳至少2個utf-8編碼的txt文件。文件格式要求：“出版年-《書名》”或“出版年-XX《書名》XX”之類，衹要出版年後用橫杠連接，有書名號括住的書名即可。例如“1601-《（萬曆）江華縣志4卷》”。")
    book_dic = {}
    for uploaded_file in uploaded_files:
        # 讀取上傳的txt文件
        txt_content = io.TextIOWrapper(uploaded_file, encoding='utf-8').read().lstrip('\ufeff')
        sentences = split_sentences(txt_content)

        file_name = os.path.splitext(uploaded_file.name)[0]
        year = file_name.split('-')[0]
        pattern = r'《(.*?)》'
        match = re.findall(pattern, file_name)
        book_name = match[0]
        book_name = year + '年《' + book_name + '》'
        year = int(year)

        book_dic[book_name] = (year, sentences)
    return book_dic

def content_source(book_dic,output_folder_path):
    '''
    功能：對輸入的書籍進行內容來源分析。對於每一本書，將它與在它之前出版的書進行比對，
          查找它的每個句子在其他書中是否有相似的句子，如果有多個，就保留其中最早的來源。
    book_dic:{book_name:(year_int,sentence_list}
    output_folder_path:輸出文件夾所在地址
    return:除了時代最早的書以外，所有的書籍內容來源xlsx表格，表結構是df[['原文', '最早的相似句', '出處']]
    '''
    min_year = min(book_dic.values(), key=lambda x: x[0])[0]
    for k, v in book_dic.items():
        year, sentences = v
        if year == min_year:
            continue
        st.write(f'正在處理{k}...')
        df = pd.DataFrame(columns=['原文', '最早的相似句', '出處'])
        for i, sentence in enumerate(sentences):
            df.at[i,'原文'] = sentence
            similar_sentences = []
            sources = []
            similar_sentence_years = []
            for k2, v2 in book_dic.items():
                year2, sentences2 = v2
                if year > year2:
                    for sentence2 in sentences2:
                        condition = is_similar(sentence,sentence2)
                        if  condition:
                            similar_sentences.append(sentence2)
                            sources.append(k2)
                            similar_sentence_years.append(year2)
            if similar_sentence_years:
                min_year_index = similar_sentence_years.index(min(similar_sentence_years))
                df.at[i, '最早的相似句'] = similar_sentences[min_year_index]
                df.at[i,'出處'] = sources[min_year_index]
        file_name = k + '.xlsx'
        out_put_file_path = os.path.join(output_folder_path, file_name)
        df.to_excel(out_put_file_path, index=False)
    st.write(f'結果已導出在“{output_folder_path}”中。')

def version_analysis(xlsx_path):
    '''
    功能：通過對content_source()導出的文本內容來源xlsx文件進行版本分析。
          計算該書中有多少比例的內容來自於其他之前的書籍，並按時間順序繪製餅狀圖。
    '''
    df = pd.read_excel(xlsx_path)
    df['原文'] = df['原文'].apply(remove_punctuation)
    df['最早的相似句'] = df['最早的相似句'].apply(remove_punctuation)
    df['相似長度'] = df.apply(lambda x: len(similar_char_index(x['原文'], x['最早的相似句'])) if pd.notna(x['最早的相似句']) else 0, axis=1)
    total_characters = df['原文'].apply(len).sum()

    result_df = df.groupby('出處')['相似長度'].sum().reset_index()
    result_df['相似字符佔原文比例'] = result_df['相似長度'] / total_characters
    result_df = result_df[result_df['相似字符佔原文比例'] != 0]
    result_dict = dict(zip(result_df['出處'],result_df['相似字符佔原文比例']))
    self_proportion = 1 - result_df['相似字符佔原文比例'].sum()
    result_dict['自身'] = self_proportion

    # 繪圖
    # 設置中文字體
    font_path = 'SimHei.ttf'
    font_prop = FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()

    labels = list(result_dict.keys())
    values = list(result_dict.values())
    percentages = [val*100 for val in values]

    fig, ax = plt.subplots(figsize=(6, 6)) 
    ax.pie(values, startangle=90, counterclock=False, wedgeprops=dict(width=0.4))

    ax.axis('equal')
    book_name = os.path.split(xlsx_path)[1].split('.')[0]
    ax.set_title(f'{book_name}來源')

    legend_labels = ['{} - {:.2f}%'.format(label, percentage) for label, percentage in zip(labels, percentages)]
    ax.legend(title="各書所佔百分比", loc='center', bbox_to_anchor=(0.5, 0.5), labels=legend_labels)
    image_path = os.path.join(os.path.dirname(xlsx_path), book_name + '.png')
    plt.savefig(image_path, bbox_inches='tight')
    st.pyplot(fig)

def version_analysis_show():
    st.subheader("版本分析")
    book_dic = split_sentences_plus()
    output_folder_path = st.text_input("請輸入分析結果導出的文件夾地址（按Enter執行），此文件夾需是空文件夾：")
    if output_folder_path:
        content_source(book_dic,output_folder_path) 
        for file_name in os.listdir(output_folder_path):
            if file_name.endswith('.xlsx'):
                file_path = os.path.join(output_folder_path, file_name)
                version_analysis(file_path)

# 主函數
def main():
    st.title("古籍對勘與版本分析")

    page = st.selectbox("選擇功能", ["文本對勘", "版本分析"])

    # 根據選擇調用函數
    if page == "文本對勘":
        text_review()
    elif page == "版本分析":
        version_analysis_show()

if __name__ == "__main__":
    main()

