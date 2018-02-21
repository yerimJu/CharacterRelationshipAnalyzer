import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from collections import *
from operator import itemgetter
import mglearn
from konlpy.tag import *
import csv
import codecs
import re
import time
import json
import numpy
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)



class KeywordExtractor:
    # 단어의 최소 등장 빈도의 기본값=5, 최대 길이의 기본값=10
    def __init__(self, min_count=5, max_length=10):
        self.min_count = min_count  # 단어 최소 길이
        self.max_length = max_length  # 단어 최대 길이
        self.sum_weight = 1  # 토큰의 개수
        self.vocabulary = {}  # 토큰 사전
        self.index2vocab = []  # 인덱스 사전

    # 토큰을 생성하는 함수
    def scan_vocabs(self, docs, verbose=True):
        # 어휘 사전 초기화
        self.vocabulary = {}

        # 상세한 옵션 선택 시 출력
        #if verbose:
        #    print('scan vocabs ... ')

        # 토큰을 카운터할 dic
        counter = {}

        # 문장을 설정한 최대 단어 길이만큼 토큰화
        for doc in docs:
            for token in doc.split():
                len_token = len(token)
                # 토큰의 개수를 카운트, dic에 없는 토큰이었으면 counter dictionary에 새로 추가하고 개수 카운트
                counter[(token, 'L')] = counter.get((token, 'L'), 0) + 1

                # 단어 최대 길이로 설정한 것까지 토큰을 분할하는 작업
                for e in range(1, min(len(token), self.max_length)):
                    if (len_token - e) > self.max_length:
                        continue

                    # 토큰을 왼쪽 부분과 오른쪽 부분으로 나누어 counter dic에 새로 추가
                    l_sub = (token[:e], 'L')
                    r_sub = (token[e:], 'R')
                    counter[l_sub] = counter.get(l_sub, 0) + 1
                    counter[r_sub] = counter.get(r_sub, 0) + 1

        # min_count 이상 출현하는 것만으로 dic 필터링
        counter = {token: freq for token, freq in counter.items() if freq >= self.min_count}

        # 토큰을 vocabulary에 순서대로 번호를 매기며 저장
        for token, _ in sorted(counter.items(), key=lambda x: x[1], reverse=True):
            self.vocabulary[token] = len(self.vocabulary)

        # 오름차순으로 정렬된 토큰 리스트 생성
        self._build_index2vocab()

        # verbose 옵션을 주면 counter dic에 들어있던 토큰의 개수를 출력한다.
        #if verbose:
        #    print('num vocabs = %d' % len(counter))
        return counter

    # 오름차순으로 정렬된 토큰 리스트를 생성하는 함수
    def _build_index2vocab(self):
        # 토큰을 vocab(key), index(value)에 저장하고 vocab으로만 된 리스트 생성
        self.index2vocab = [vocab for vocab, index in sorted(self.vocabulary.items(), key=lambda x: x[1])]
        # 토큰의 개수
        self.sum_weight = len(self.index2vocab)

    def extract(self, docs, beta=0.85, max_iter=10, verbose=True, vocabulary=None, bias=None, rset=None):
        rank, graph = self.train(docs, beta, max_iter, verbose, vocabulary, bias)

        lset = {self.int2token(idx)[0]: r for idx, r in rank.items() if self.int2token(idx)[1] == 'L'}
        if not rset:
            rset = {self.int2token(idx)[0]: r for idx, r in rank.items() if self.int2token(idx)[1] == 'R'}

        # 후처리 (조사 제외 단어 추출, 합성어 처리)
        keywords = self._select_keywords(lset, rset)
        keywords = self._filter_compounds(keywords)
        keywords = self._filter_subtokens(keywords)

        return keywords, rank, graph

    # 조사를 포함한 어절을 조사를 제외하고 추출되도록 필터링
    def _select_keywords(self, lset, rset):
        keywords = {}

        for word, r in sorted(lset.items(), key=lambda x: x[1], reverse=True):

            len_word = len(word)
            # 한 글자로 된 것은 제외
            if len_word == 1:
                continue

            is_compound = False
            # 어절을 분리해 왼쪽에 있는 것이 키워드에 들어있으면서 rset에 포함되어 있으면 결합어 is_compound=True
            for e in range(2, len_word):
                if (word[:e] in keywords) and (word[:e] in rset):
                    is_compound = True
                    break

            if not is_compound:
                keywords[word] = r

        return keywords

    # 합성어 필터링 ('A', 'B'가 'AB'보다 rank가 높을 경우 AB는 합성어로 보아 필터링)
    def _filter_compounds(self, keywords):
        keywords_ = {}
        for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
            len_word = len(word)

            if len_word <= 2:
                keywords_[word] = r
                continue

            if len_word == 3:
                if word[:2] in keywords_:
                    continue

            is_compound = False
            for e in range(2, len_word - 1):
                if (word[:e] in keywords) and (word[:e] in keywords):
                    is_compound = True
                    break

            if not is_compound:
                keywords_[word] = r

        return keywords_

    # 한 글자의 rank가 높으므로 substring 필터링 (가령 'ABC'가 추출되면 'AB'도 추출될 가능성이 있음
    def _filter_subtokens(self, keywords):
        subtokens = set()
        keywords_ = {}

        for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
            subs = {word[:e] for e in range(2, len(word) + 1)}

            is_subtoken = False
            for sub in subs:
                if sub in subtokens:
                    is_subtoken = True
                    break

            if not is_subtoken:
                keywords_[word] = r
                subtokens.update(subs)

        return keywords_

    def train(self, docs, beta=0.85, max_iter=10, verbose=True, vocabulary=None, bias=None):
        # 입력된 토큰 사전과 구축된 토큰 사전이 없는 경우 새로 생성
        # 모든 어절에 대해서 L과 R을 구분하여 최소 등장 빈도를 고려한 dic을 구축
        if (not vocabulary) and (not self.vocabulary):
            self.scan_vocabs(docs, verbose)
        # 구축된 토큰 사전이 있으면 사용
        elif (not vocabulary):
            self.vocabulary = vocabulary
            self._build_index2vocab()

        # bias를 입력하지 않았으면 dic 형식으로 선언
        if not bias:
            bias = {}

        # 문서를 토큰화 하고 토큰과 카운터값을 dic으로 만든 것을 그래프로 생성
        graph = self._construct_word_graph(docs)

        dw = self.sum_weight / len(self.vocabulary)
        rank = {node: dw for node in graph.keys()}

        for num_iter in range(1, max_iter + 1):
            rank = self._update(rank, graph, bias, dw, beta)
            #sys.stdout.write('\riter = %d' % num_iter)

        return rank, graph

    # 토큰에 해당하는 value를 리턴, 토큰 사전에 없으면 -1 리턴
    def token2int(self, token):
        return self.vocabulary.get(token, -1)

    # index가 토큰 어휘 사전의 개수 내에서 접근이 이루어지면 그 값을 반환하고 개수를 벗어나면 None 반환
    def int2token(self, index):
        return self.index2vocab[index] if (0 <= index < len(self.index2vocab)) else None

    def _construct_word_graph(self, docs):
        def normalize(graph):
            graph_ = defaultdict(lambda: defaultdict(lambda: 0))
            for from_, to_dict in graph.items():
                sum_ = sum(to_dict.values())
                for to_, w in to_dict.items():
                    graph_[to_][from_] = w / sum_
            return graph_

        # 2차원 트리 구조의 dic 생성
        graph = defaultdict(lambda: defaultdict(lambda: 0))

        # docs는 문장 하나씩을 원소로 가지는 리스트
        for doc in docs:

            # 어절 단위로 문장을 분리해 리스트 생성
            tokens = doc.split()

            # 분리할 토큰이 없으면 다음 문장에 대해서 실행
            if not tokens:
                continue

            # 각각의 토큰을 의미를 가지는 L과 R로 새로이 분리해서 links에 추가
            links = []
            for token in tokens:
                links += self._intra_link(token)

            # 외부 토큰과의 관계
            if len(tokens) > 1:
                tokens = [tokens[-1]] + tokens + [tokens[0]]
                links += self._inter_link(tokens)

            # 토큰 분할 사전에 들어있는 조합만 리턴
            links = self._check_token(links)
            if not links:
                continue

            # 토큰의 카운터 값을 찾아서 리턴
            links = self._encode_token(links)
            for l_node, r_node in links:
                graph[l_node][r_node] += 1
                graph[r_node][l_node] += 1

        graph = normalize(graph)
        return graph

    # 하나의 토큰 내에서 L과 R을 분리하는 작업 (내부)
    # ex) tokens = ['오늘도', '달린다']
    # [(('오', 'L'), ('늘도', 'R')), (('오늘', 'L'), ('도', 'R')), (('달', 'L'), ('린다', 'R')), (('달린', 'L'), ('다', 'R'))]
    def _intra_link(self, token):
        links = []
        len_token = len(token)
        for e in range(1, min(len_token, 10)):
            if (len_token - e) > self.max_length:
                continue
            links.append(((token[:e], 'L'), (token[e:], 'R')))
        return links

    # 선택된 토큰과 왼쪽, 오른쪽에 존재하는 토큰을 분할한 토큰과의 관계 리턴(외부)
    # ex) tokens = ['오늘도', '달린다'] -> 원형 큐처럼 변형 후 토큰 계산 -> ['달린다', '오늘도', '달린다', '오늘도']
    # [(('다', 'R'), ('오늘도', 'L')), (('린다', 'R'), ('오늘도', 'L')),
    # (('오늘도', 'L'), ('달', 'L')), (('오늘도', 'L'), ('달린', 'L')),
    # (('도', 'R'), ('달린다', 'L')), (('늘도', 'R'), ('달린다', 'L')),
    # (('달린다', 'L'), ('오', 'L')), (('달린다', 'L'), ('오늘', 'L'))]
    def _inter_link(self, tokens):
        def rsub_to_token(t_left, t_curr):
            return [((t_left[-b:], 'R'), (t_curr, 'L')) for b in range(1, min(10, len(t_left)))]

        def token_to_lsub(t_curr, t_rigt):
            return [((t_curr, 'L'), (t_rigt[:e], 'L')) for e in range(1, min(10, len(t_rigt)))]

        links = []
        for i in range(1, len(tokens) - 1):
            links += rsub_to_token(tokens[i - 1], tokens[i])
            links += token_to_lsub(tokens[i], tokens[i + 1])
        return links

    # 토큰 분할 사전에 들어있는 조합만 리턴
    def _check_token(self, token_list):
        return [(token[0], token[1]) for token in token_list if
                (token[0] in self.vocabulary and token[1] in self.vocabulary)]

    # 토큰의 카운터 값을 찾아서 리턴
    def _encode_token(self, token_list):
        return [(self.vocabulary[token[0]], self.vocabulary[token[1]]) for token in token_list]

    def _update(self, rank, graph, bias, dw, beta):
        rank_new = {}
        for to_node, from_dict in graph.items():
            rank_new[to_node] = sum([w * rank[from_node] for from_node, w in from_dict.items()])
            rank_new[to_node] = beta * rank_new[to_node] + (1 - beta) * bias.get(to_node, dw)
        return rank_new


def KR_Word_Rank(file_name, min_count, max_length, beta=0.85, max_iter=7, verbose=True):
    # min_count : 단어의 최소 출현 빈도수 (그래프 생성 시)
    # max_length : 단어의 최대 길이
    # beta : PageRank의 decaying factor beta

    texts = open(file_name, 'r').readlines()
    wordrank_extractor = KeywordExtractor(min_count, max_length)
    keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter, verbose)

    word_list_key = []
    word_list_val = []
    for key, val in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:300]:
        word_list_key.append(key)
        word_list_val.append(val)
    return word_list_key, word_list_val


def Entity_Recognition_On_Freq(file_name, full_file_name):
    # 모든 단어로 set을 만들고 조사를 하나씩 빼면서 set을 생성
    f = open(file_name, 'r')
    texts = re.sub('[-=.#\'\"/!?:$(){},\n\t]', ' ', f.read().replace('\n', ''))
    full_f = open(full_file_name, 'r')
    full_texts = re.sub('[-=.#\'\"/!?:$(){},\n\t]', ' ', full_f.read().replace('\n', ''))

    # 어절 단위로 글자 수가 3개 이상인 것만 추려냄 (찾고자 하는 명사는 최소 2자 + 조사라고 가정)
    # 명백하게 개체명이 될 수 없는 어절은 필터링
    h_filter_POS = ['mag', 'nbu', 'npd', 'ncpa']  # Hannanum 분석기 필터링
    k_filter_POS = ['VV', 'VA', 'NNB', 'MDT', 'MAG', 'XR']  # Kkma 분석기 필터링
    t_filter_POS = ['Adjective', 'Adverb', 'Verb', 'VerbPrefix', 'Conjunction']  # twitter 분석기 필터링

    # 각 분석기에서 정확하게 판별 가능한 형태소들을 필터링
    word_set = [i for i in texts.split() if (len(i) >= 3)
                and h.analyze(i)[0][0][0][1] not in h_filter_POS
                and t.pos(i)[0][1] not in t_filter_POS
                and k.pos(i)[0][1] not in k_filter_POS]
    len_w_set = len(word_set)

    # 기존 word_set 토큰에 대해서 2글자가 될 때까지 오른쪽에서부터 하나씩 슬라이싱 해서 word_set에 추가
    # 숫자로만 된 것 필터링
    for i in range(len_w_set):
        word_set += [word_set[i][:-j] for j in range(1, len(word_set[i]) - 1) if not str.isdigit(word_set[i][:-j])]

    # 중복 제거
    word_set = list(set(word_set))
    # 토큰을 카운트할 토크 카운트 행렬을 생성
    word_count = [0 for _ in range(len(word_set))]

    # word_set의 토큰별로 texts의 왼쪽 의미 부분에 등장하는 횟수를 카운트
    for i in texts.split():
        for j in word_set:
            if j == i[:len(j)]:
                try:
                    word_count[word_set.index(j)] += 1
                except:
                    pass

    # temp는 '찾을 단어 : 바꿀 단어'로 된 dic, view는 '바꿀 단어 : count 값'으로 된 dic
    temp = {}
    view = {}

    for i in word_set:
        try:
            # 토큰마다 잘게 잘린 토큰이 해당 토큰보다 더 많은 빈도 수를 가진 경우
            # 그 토큰은 조사가 연결된 것으로 보아 조사를 제거한 것으로 바꿀 목록에 저장
            count = [word_count[word_set.index(i)]]
            count += [word_count[word_set.index(i[:-j])] for j in range(1, len(i) - 1)]

            max_idx = count.index(max(count))

            # 바꿀 단어가 문장에서 단어+' '꼴의 비율이 높으면 등장인물이 아닐 것이므로 필터링
            space_ratio = texts.count(i[:-max_idx] + ' ') / word_count[word_set.index(i[:-max_idx])]
            if space_ratio > 0.5:
                continue

            if max_idx != 0:
                temp[i] = i[:-max_idx]
                view[i[:-max_idx]] = max(count)
        except:
            pass

    # 빈도 내림차순으로 대명사를 필터링한 리스트 생성
    view_list = [i for i in sorted(view.items(), key=itemgetter(1), reverse=True) if k.pos(i[0])[0][1] != 'NP']
    view_list_key = []
    view_list_val = []

    for i in view_list:
        # 등장인물들 뒤에 절대 붙지 않는 단어인 경우 필터링
        if any([i[0] + k in full_texts for k in ['들', '었', '할', '함', '합', '해', '했']]):
            continue

        if i[0] in ['사실', '있었', '그리', '자신', '얼굴', '지금', '아무', '아주', '대로', '그때', '정말', '조금', '처음']:
            continue

        # '에'가 붙는 경우
        if (i[0] + '에 ' in full_texts) and \
                (not any(i[0] + k in full_texts for k in ['에 대', '에 관', '에 의'])):
            continue
        view_list_key.append(i[0])
        view_list_val.append(i[1])
    # 단어와 단어 cnt 값 출력
    # for i in range(len(view_list_key)):
    #    print(view_list_key[i], view_list_val[i])
    return view_list_key, view_list_val


def clustering_xy_data(x_key, x_val, y_key, y_val, dir='R'):
    data_name = []
    data_val = []

    if dir == 'R':
        # erof 기반으로 묶음
        for i in y_key:
            data_name.append(i)
            data_val.append([y_val[y_key.index(i)],
                             x_val[x_key.index(i)] if i in x_key else 0])

    elif dir == 'L':
        # X는 krwordrank 기반으로 묶음
        for i in x_key:
            data_name.append(i)
            data_val.append([x_val[x_key.index(i)],
                             y_val[y_key.index(i)] if i in y_key else 0])
    else:
        # x, y를 모두 erof로 두고 묶음
        for i in y_key:
            data_name.append(i)
            data_val.append([y_val[y_key.index(i)],
                             y_val[y_key.index(i)]])

    return data_name, np.float32(np.array(data_val))


def S_Scaler(xy_data):
    scaler = StandardScaler()
    scaler.fit(xy_data)
    return scaler.transform(xy_data)


def DBSCAN_show(data_name, xy_data, show='n'):
    xy_data_scaled = S_Scaler(xy_data)

    dbscan = DBSCAN()
    clusters = dbscan.fit_predict(xy_data_scaled)
    # print("클러스터 레이블:\n{}".format(clusters))
    plt.scatter(xy_data_scaled[:, 0], xy_data_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=20,
                edgecolors='black')
    for i in range(len(data_name)):
        plt.annotate(data_name[i], xy=(xy_data_scaled[i][0], xy_data_scaled[i][1]),
                     xytext=(xy_data_scaled[i][0], xy_data_scaled[i][1] + 0.1))
    if show == 'y':
        plt.show()


def Agglomerative(data_name, xy_data, Num_of_Cluster=6, cluster_select_rate=0.5, d_from_origin=0.4, show='n'):
    xy_data_scaled = S_Scaler(xy_data)

    agg = AgglomerativeClustering(n_clusters=Num_of_Cluster)
    assignment = agg.fit_predict(xy_data_scaled)

    plt.scatter(xy_data_scaled[:, 0], xy_data_scaled[:, 1], c=assignment, s=20, edgecolors='black')

    for i in range(len(data_name)):
        plt.annotate(data_name[i], xy=(xy_data_scaled[i][0], xy_data_scaled[i][1]),
                     xytext=(xy_data_scaled[i][0], xy_data_scaled[i][1] + 0.1))
    if show == 'y':
        plt.show()


def KMeans_show(data_name, xy_data, Num_of_Cluster=6, cluster_select_rate=0.5, d_from_origin=0.4, show='n'):
    xy_data_scaled = S_Scaler(xy_data)

    if len(xy_data_scaled) < Num_of_Cluster :
        Num_of_Cluster = len(xy_data_scaled)

    kmeans = KMeans(n_clusters=Num_of_Cluster)
    kmeans.fit(xy_data_scaled)
    # print("클러스터 레이블:\n{}".format(kmeans.labels_))

    # 데이터 보여주기
    assignments = kmeans.labels_
    plt.scatter(xy_data_scaled[:, 0], xy_data_scaled[:, 1], c=assignments, s=20, edgecolors='black')

    # 클러스터 중심 보여주기
    # print('centers : ', kmeans.cluster_centers_)

    # k_means 클러스터 중심 표시
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c=[i for i in range(len(kmeans.cluster_centers_))], s=8000, marker='o', alpha=0.1)

    # data_name 그래프에 표시
    for i in range(len(data_name)):
        plt.annotate(data_name[i], xy=(xy_data_scaled[i][0], xy_data_scaled[i][1]),
                     xytext=(xy_data_scaled[i][0], xy_data_scaled[i][1] + 0.1))
    if show == 'y':
        plt.show()

    # 클러스터 번호마다 클러스터의 중심과 (0, 0) 사이의 거리 계산
    distance = {}
    for i in range(len(kmeans.cluster_centers_)):
        distance[i] = np.sqrt(sum([kmeans.cluster_centers_[i][j] ** 2 for j in range(2)]))

    # 클러스터링
    sort_distance = [i for i in sorted(distance.items(), key=itemgetter(1), reverse=True)]
    selected_cluster = [sort_distance[i][0] for i in range(int((len(sort_distance) + 1) * cluster_select_rate))
                        if sort_distance[i][1] > np.sqrt(max(xy_data_scaled[:, 0]) ** 2 +
                                                         max(xy_data_scaled[:, 1] ** 2)) * d_from_origin]

    # 모든 단어에 대해서 selected_cluster에 포함된 단어를 result에 추가해서 리턴
    result = []
    for i in range(len(data_name)):
        if assignments[i] in selected_cluster:
            result.append(data_name[i])
    print('Selected nouns : ', result)
    return result


def slice_file(file_name, file_include_rate=0.8):
    f = open(file_name + '.txt', 'r')
    texts = f.read()
    texts_len = len(texts)
    # 소설이 2만 자 미만인 경우 2개로 분리, 2만 자보다 긴 경우 2만 자씩 분리
    Num_of_pieces = (2 if texts_len < 20000 else int(texts_len / 20000) + 1)

    # 2만 자가 넘는 소설을 분리했을 때 마지막 구간이 2만 자가 안 되면 그 구간은 집계에서 제외
    if texts_len > 20000 and (len(texts) % 20000) / 20000 < file_include_rate:
        Num_of_pieces -= 1

    batch_size = (int(texts_len / 2) + 1 if texts_len < 20000 else 20000)

    # 분리할 자수별로 번호+_slice라는 이름의 개별 파일 생성
    for i in range(Num_of_pieces):
        with open(file_name + str(i) + '_slice.txt', 'w') as temp_f:
            temp_f.write(texts[i * batch_size:(i + 1) * batch_size])
    return Num_of_pieces


'''
#StandardScaler 평균 0, 분산 1로 Scaler
def StandardScaler(plist):
    mean_on_plist = np.sum(plist, axis=0) / len(plist)
    std_on_plist = np.sqrt(np.sum((np.array(plist) - mean_on_plist) ** 2, axis=0)) / len(plist)
    plist_scaled = (np.array(plist) - mean_on_plist) / std_on_plist
    return plist_scaled
'''
'''
#출력
for i in range(len(view_list) if len(view_list) < 20 else 40):
    print(i, '%2d  ' % (i+1), view_list[i][1], view_list[i][0])
exit(1)
for i in range(0):
    print("h : ", h.analyze(view_list[i][0]))
    print("k : ", k.pos(view_list[i][0]))
    print("t : ", t.pos(view_list[i][0]), "\n")
'''

def load_kosac():
    global dic
    with codecs.open("input/polarity.csv", "r", "utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        dic = dict()
        for row in reader:
            # 긍정 / 부정 단어만 감성 사전에 추가

            if row.get("max.value") == "POS" or row.get("max.value") == "NEG":
                key = str()
                str1 = row.get("ngram")
                str1_list = str1.split(";")

                # 뜻을 가진 문구를 제외한 한 어절은 감성 사전에 추가하지 않음
                if len(str1_list) == 1 and not (key.endswith("/N") or key.endswith("/V") or key.endswith("/M")):
                    continue

                for s in str1_list:
                    str2 = str(s).split("/")
                    # 정규식 표현과 혼동되는 표현 수정
                    str2[0] = str2[0].replace("*", "")
                    key += str2[0] + "/" + str2[1][0]

                value = row.get("max.value") + " " + row.get("max.prop")
                mydict = {key: value}
                dic.update(mydict)

    # key를 기준으로 정렬
    # dic = sorted(dic.items(), key=operator.itemgetter(0))
    # print("Sentiment Dictionary : ", dic)
    print("Dictionary size : ", len(dic))


def divide_morpheme(stnce):
    divided_result = str()
    morpheme_list = kkma.pos(stnce)
    for morpheme in morpheme_list:
        divided_result += morpheme[0] + '/' + morpheme[1][0]

    # '가ㄹ'이 '갈ㄹ'로 나오는 현상 수정
    divided_result = divided_result.replace("갈/Vㄹ/E", "가/Vㄹ/E")

    return divided_result


def calc_senti_num(senti_num, values, const):
    senti_kind_value = float(values[1])

    if values[0] == "POS":
        return senti_num + (const * senti_kind_value)
    elif values[0] == "NEG":
        return senti_num - (const * senti_kind_value)
    else:
        return senti_num


def character_to_integer(temp_list, char_list):
    for i in range(len(temp_list)):
        for j in range(len(char_list)):
            if temp_list[i] == char_list[j]:
                temp_list[i] = j

    return temp_list


def analyze_polarity(novel, character_list):
    global kkma
    kkma = Kkma()

    novel = novel.replace("\"", "").replace("\'", "").replace("\n", "")
    sentence_list = re.split("[.?!]+\s+", novel)

    # 각 문장에서 주인공이 등장하면 리스트에 추가
    # [[sentence, [등장인물1, 2 ..]], [] ...]
    char_sentence_list = []

    for sentence in sentence_list:
        is_first_sentence = False
        for char in character_list:
            if re.search(char, sentence):
                if not is_first_sentence:
                    new_sentence = sentence.replace(char, "")
                    char_sentence_list.append([new_sentence, [char]])
                    is_first_sentence = True
                else:
                    last_cslist = char_sentence_list[len(char_sentence_list) - 1]
                    last_cslist[0] = last_cslist[0].replace(char, "")
                    last_cslist[1].append(char)

    '''print("before remove ---", len(char_sentence_list))
    for sen in char_sentence_list:
        print(sen)'''

    # 인물이 1개인 경우 삭제
    new_char_sentence_list = list()
    for obj in char_sentence_list:
        if len(obj[1]) > 1:
            new_char_sentence_list.append(obj)
    char_sentence_list = new_char_sentence_list

    '''print("after remove ---", len(char_sentence_list))
    for sen in char_sentence_list:
        print(sen)'''

    # 문장에 대한 감성 분석
    # 한 문장에 대해서 사전의 모든 key 조사
    for index in range(len(char_sentence_list)):
        morpheme_list = divide_morpheme(char_sentence_list[index][0])
        '''print('\nConverted sentence : ', morpheme_list)
        print('The sentiment of this sentence', end=" : ")'''
        senti_num = 0.0
        temp_indices = []
        temp_keys = []

        for key in sorted(dic.keys()):
            sentiment = re.search(key, morpheme_list)

            # 문장에 현재 key가 매칭되지 않을 경우 넘어감
            if not sentiment:
                continue

            is_new = True
            values = str(dic[key]).split()

            # 기존 key들에 대한 현재 key의 유효성 검사
            for i in range(len(temp_keys)):

                is_parent = re.search(key, temp_keys[i])
                is_child = re.search(temp_keys[i], key)

                # key가 기존 key들의 상위 개념일 경우(포괄적)
                # - 무시 > 다음 key로 넘어감
                if is_parent:
                    is_new = False
                # key가 기존 key들의 하위 개념이면서 시작 위치도 같을 경우(구체적)
                # - 중복되는 기존 key 삭제 후 현재 key 추가
                # 위치가 같지 않으면 새로운 key로 취급
                elif is_child and sentiment.pos == temp_indices[i]:
                    is_new = False
                    temp_values = str(dic[temp_keys[i]]).split()
                    senti_num = calc_senti_num(senti_num, temp_values, 1)
                    temp_keys.pop(i)
                    temp_indices.pop(i)

                    constant = len(re.findall(key, morpheme_list))
                    senti_num = calc_senti_num(senti_num, values, constant)
                    temp_keys.append(key)
                    temp_indices.append(sentiment.pos)

            # temp_key의 유효성 검사를 통과했다면 새로운 key이므로 추가해줌
            if is_new:
                # 추가하려는 단어가 여러 번 나왔을 경우, 강조
                # 참고 : 2번째 단어는 index를 알 수 없음
                constant = len(re.findall(key, morpheme_list))
                senti_num = calc_senti_num(senti_num, values, constant)
                temp_keys.append(key)
                temp_indices.append(sentiment.pos)

        senti_num = round(senti_num, 2)
        # 한 문장의 감성 지수 최대 상한선은 ±5
        if senti_num > 5.0:
            senti_num = 5.0
        elif senti_num < -5.0:
            senti_num = -5.0

        '''print(senti_num)
        print(temp_keys)

        if senti_num > 0:
            print('This sentence is positive!')
        elif senti_num == 0:
            print('This sentence is neural.')
        else:
            print('This sentence is negative!')
        print()'''

        char_sentence_list[index].append(senti_num)

    for s in char_sentence_list:
        print(s)
    print()

    # 각 주인공 : 다른 주인공들에 관한 matrix 만들기
    # 3차원 matrix 생성.
    # k번째 주인공, col은 각 문장, row는 각 등장인물
    sen_size = len(char_sentence_list)
    char_size = len(character_list)

    matrix = [[[0 for col in range(sen_size + 1)] for row in range(char_size)] for k in range(char_size)]

    for k in range(char_size):
        for i in range(char_size):
            matrix[k][i][0] = character_list[i]

    # 각 sentence 별로, 각 등장인물에 대한 matrix 생성
    count = 1
    while count <= sen_size:

        sen = char_sentence_list[count - 1]

        int_character_list = character_to_integer(sen[1], character_list)

        for main in int_character_list:
            if len(sen[1]) <= 1:
                continue

            for sub in int_character_list:
                if main is not sub:
                    matrix[main][sub][count] = sen[2]

        count = count + 1

    for k in range(char_size):
        print(character_list[k], "matrix 최종 상태 : ")
        for i in range(char_size):
            print(matrix[k][i])

    print()
    link_list = list()
    for k in range(char_size):
        sum_k = [round(sum(i[1:]), 2) for i in matrix[k]]
        print(sum_k)

        for i in range(char_size):
            if k == i:
                continue

            print(character_list[k], "and", character_list[i], "displayed a ", end="")
            if sum_k[i] > 0:
                print("positive relationship :)")
                link_list.append([character_list[k], character_list[i], sum_k[i]])
            elif sum_k[i] == 0:
                print("no relationship :|")
            else:
                print("negative relationship :(")
                link_list.append([character_list[k], character_list[i], sum_k[i]])
        print()

    write_json_file(character_list, link_list, matrix)


# parse lists to json files
def write_json_file(character_list, link_list, character_matrix):

    # make JSON dumps
    file_data = OrderedDict()
    file_data["nodes"] = list()
    file_data["links"] = list()

    for i in range(len(character_list)):
        file_data["nodes"].append({"id": character_list[i], "group": i})

    for i in range(len(link_list)):
        file_data["links"].append({"source": link_list[i][0], "target": link_list[i][1], "value": link_list[i][2]})

    print(json.dumps(file_data, ensure_ascii=False, indent="\t"))

    # write JSON file
    '''with open("result.json", "w", encoding="utf-8") as make_file:
        json.dump(file_data, make_file, ensure_ascii=False, indent="\t")'''
    with open("C:\inetpub/wwwroot/capstone/result_" + file_name.split("/")[1] +".json", "w", encoding="utf-8") as make_file:
        json.dump(file_data, make_file, ensure_ascii=False, indent="\t")

    # write TSV file
    '''for number in range(len(character_list)):

        cur_char = character_list[number]

        with open("C:\inetpub/wwwroot/capstone/resource2/" + str(number) + ".tsv", "w", encoding="utf-8") as make_file:
            # write header
            make_file.write("date\t")
            for temp_char in character_list:
                if cur_char == temp_char:
                    continue
                make_file.write(temp_char + "\t")
            make_file.write("\n")

            # write numbers
            mat = numpy.transpose(character_matrix[number])
            print(mat)
            for row_num in range(len(mat)):
                if row_num == 0:
                    continue

                make_file.write(str(row_num) + "\t")

                col_num = 0
                for col in mat[row_num]:
                    col_num = col_num + 1
                    if str(col) == '0':
                        make_file.write("\t")
                        continue
                    make_file.write(str(col))
                    if col_num != len(mat[row_num])-1:
                        make_file.write("\t")

                make_file.write("\n")'''


if __name__ == '__main__':
    start_time = time.time()

    h = Hannanum()
    k = Kkma()
    t = Twitter()

    global file_name
    file_name = 'input/파천무'
    #file_name = '고등어'
    #file_name = '딱딱산활활산'
    #file_name = '개구리왕자'

    #file_name = 'SKT'
    #file_name = '무림에서왔수다'
    #file_name = '황제의검'
    #file_name = '표사'
    #file_name = '삼절삼괴'
    #file_name = '천검월야행'
    #file_name = '초애몽'
    #file_name = '파천무'


    print('Document Len : ', len(open(file_name+'.txt', 'r').read()))

    # parameter
    file_include_rate = 0.8     #파일 분할 시 마지막 파일을 포함시킬지를 결정하는 내용의 비율
    min_count = 3               #단어의 최소 등장 횟수
    max_length = 10             #단어의 최대 길이
    clustering_dir = 'R'        #클러스터링 기준 x 결정 (L은 krwordrank 기준, R은 EROF 기준, LR은 둘 다 EROF 기준)
    Num_of_Cluster = 6          #클러스터링 개수
    cluster_select_rate = 0.5   #클러스터링 선택 비율
    d_from_origin = 0.35        #가장 먼 클러스터 중심 거리를 기준으로 결과에 포함시킬 클러스터 중심의 상대적 거리
    graph_show = 'n'            #그래프의 표시 유무

    #파일이 너무 긴 경우 batch_size만큼씩 분할
    print("file name : ", file_name)
    Num_of_pieces = slice_file(file_name, file_include_rate = file_include_rate)
    print('Num of pieces : ', Num_of_pieces, '\n')

    #최종 출력 value list
    character_list_result = []

    #분리된 문서들에 대해서 result를 구한 후 합산
    for i in range(Num_of_pieces):
        slice_file_name = file_name+str(i)+'_slice.txt'

        #krwordrank
        word_list_key, word_list_val = KR_Word_Rank(file_name=slice_file_name, min_count=min_count, max_length=max_length)

        #entity recognition on frequency
        view_list_key, view_list_val = Entity_Recognition_On_Freq(slice_file_name, file_name+'.txt')

        #krwordrank와 entity recognition on frequency를 shape=[None, 2]로 묶어서 X로 반환
        #dir은 어떤 값을 기준으로 2차원에 도시할 것인지 설정하는 값
        data_name, X = clustering_xy_data(word_list_key, word_list_val, view_list_key, view_list_val, dir=clustering_dir)

        #K_Means 방식으로 클러스터링하고 2차원 좌표평면에 도시화
        character_list_result += KMeans_show(data_name=data_name, xy_data=X, Num_of_Cluster=Num_of_Cluster,
                                             cluster_select_rate=cluster_select_rate, d_from_origin=d_from_origin,
                                             show=graph_show)

        #병합군집으로 클러스터링하고 2차원 좌표평면에 도시화
        #character_list_result += Agglomerative(data_name=data_name, xy_data=X, Num_of_Cluster=Num_of_Cluster,
        #                                       cluster_select_rate=cluster_select_rate, d_from_origin=d_from_origin,
        #                                       show=graph_show)

        #DBSCAN 방식으로 클러스터링하고 2차원 좌표평면에 도시화
        #DBSCAN_show(data_name, X, show=graph_show)

    # 합한 결과의 중복을 제거하고 list로 반환
    character_list_result = list(set(character_list_result))
    print('Extracted nouns : ', character_list_result)

    mid_time = time.time()
    # 감성 분석 시작
    print("\n--- Start relationship analysis ---")
    load_kosac()
    text = open(file_name + ".txt").read()
    analyze_polarity(text, character_list_result)

    print("--- preprocessing & clustering : %.2f seconds ---" % (mid_time - start_time))
    print("--- sentiment analyzing : %.2f seconds ---" % (time.time() - mid_time))
    print("--- TOTAL TIME : %.2f seconds ---" % (time.time() - start_time))
