#As of now, there's only one huge dirty file.
#Presumably to be divided into a more reasonable structure.

##EasyNMT

#Takes list of english strings.
#Returns list of latvian strings
#model_type:
#  'bad' - use model opus-mt
#  'good' - use model m2m_100_1.2B

import torch
import gc
import torch
import clip
import gzip
import html
import os
import ftfy
import regex as re

from PIL import Image
from functools import lru_cache
from easynmt import EasyNMT


easynmt_model_type = None
easynmt_model = None
def translate(texts, model_type = 'bad', batch_size = 64, progress = False):
  if progress:
    print('Translating.')
    
  global easynmt_model_type
  global easynmt_model
  global easynmt

  if progress:
    print('Getting Model')

  if easynmt_model_type != model_type:
    if model_type == 'good':
      easynmt_model = EasyNMT('m2m_100_1.2B')
    elif model_type == 'bad':
      easynmt_model = EasyNMT('opus-mt')
    else:
      raise ValueError('EasyNMT model ' + model_type + ' not supported.')
    easynmt_model_type = model_type
    gc.collect(2)
  
  if progress:
    print('Got Model')

  cut_texts = []
  for text in texts:
    cut_texts.append(text[:200])
  results = []
  for offset in range(0, len(texts), batch_size):
    if progress:
      print(offset, '/', len(texts))
    results+= easynmt_model.translate(cut_texts[offset: offset + batch_size], source_lang = 'lv', target_lang = 'en')
  gc.collect(2)

  return results
  
##CLIP

CLIP_configed = False

def config_torch():
    global torch
    torch.device('cpu')

#Get clip model
def config_CLIP():
  global context_length
  global CLIP_model
  global CLIP_configed
  global CLIP_preprocess
  global clip
  if not CLIP_configed:

    config_torch()
    CLIP_model, CLIP_preprocess = clip.load('ViT-B/32')
    CLIP_configed = True
    context_length = CLIP_model.context_length

#@title

#!pip install ftfy




@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = "bpe_simple_vocab_16e6.txt.gz"):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text


#Takes list of texts
#Returns CLIP text encodings
def clip_encode_text(texts, batch_size = 1024, progress = False):
  config_CLIP()
  tokenizer = SimpleTokenizer()
  text_tokens = [tokenizer.encode(text) for text in texts]
  text_input = torch.zeros(len(text_tokens), CLIP_model.context_length, dtype=torch.long)
  sot_token = tokenizer.encoder['<|startoftext|>']
  eot_token = tokenizer.encoder['<|endoftext|>']

  for i, tokens in enumerate(text_tokens):
      tokens = [sot_token] + tokens[:75] + [eot_token]
      text_input[i, :len(tokens)] = torch.tensor(tokens)

  #text_input = text_input.cuda()

  batch = 1024

  text_features = []

  for offset in range(0, len(texts), batch):
    if progress:
      print(offset)

    with torch.no_grad():
        text_features.append(CLIP_model.encode_text(text_input[offset: offset + batch]).float())

  text_features = torch.cat(text_features).cpu().numpy()

  return text_features

#Takes list of strings
#Returns list of CLIP text encodings
#Option:
#  load - Loads this from google drive
#  rerun - Runs CLIP on strings
#  cache - Runs CLIP on strings and saves it to google drive
#  check - Loads if file exists, caches if not
#  skip - dosen't do anything and returns None
def get_text_encodings(texts, output_name, option = 'load', progress = False):
  if progress:
    print('Getting CLIP text encodings.')
  
  clip_path = ''
    
  import pickle
  pickle_path = clip_path + output_name + 'CLIP_text_encodings.pkl' 
  result = None

  def load():
    nonlocal result
    if not os.path.isfile(pickle_path):
      raise IOError('File ' + pickle_path, ' not found.')
    pkl_file = open(pickle_path, 'rb')
    result = pickle.load(pkl_file)

  def rerun():
    nonlocal result
    result = clip_encode_text(texts, progress = progress)

  def cache():
    rerun()
    
    #Save results
    output_file = open(pickle_path, 'wb')
    pickle.dump(result, output_file)
    output_file.close()

  def check():
    try:
      load()
    except IOError:
      cache()

  if option == 'load':
    load()
  elif option == 'rerun':
    rerun()
  elif option == 'cache':
    cache()
  elif option == 'check':
    check()
  elif option == 'skip':
    return None
  else:
    raise ValueError('Option "' + option, '" doesn\'t exist.')

  return result
  
def cache_load(path):
  if not os.path.isfile(path):
    raise IOError('File ' + path + ' not found.')

  if path[-4:] == '.pkl':
    import pickle as pkl
    pkl_file = open(path, 'rb')
    result = pkl.load(pkl_file)
    pkl_file.close()
  elif path[-5:] == '.json':
    import json
    json_file = open(path, 'r')
    json_data = json_file.read()
    result = json.loads(json_data)
    json_file.close()
  else:
    raise Exception('file type not supported. File - ' + path)
  
  return result

#Saves object as pickle or json file.
#result - object to be saved
#file_type - what extension to use
def cache_save(result, path, file_type = 'json'):
  if file_type == 'pkl':
    import pickle as pkl
    pkl_file = open(path, 'wb')
    pkl.dump(result, pkl_file)
    pkl_file.close()
  elif file_type == 'json':
    import json
    json_file = open(path, 'w')
    json.dump(result, json_file)
    json_file.close()
  else:
    raise ValueError('File type ' + file_type + ' not supported.')
  return result


#Decorator for adding caching functionality to a function.
#default_option/cache_option:
#  load - loads result from disk
#  run - gets result by running the function
#  save - runs function and saves result
#  check - checks whether file exists on disk, runs and saves otherwise
#file_type:
#  pkl - pickle file
#  json - JSON file
#inherit_prefix - whether to pass argument "cache_prefix = cache_file" to function being wrapped.
#Example:
#  @cacheable(file_type = 'pkl', default_option = 'run')
#  def calc_stuff(x, y):
#    return x + y
#
#  calc_stuff(x = 5, y = 6, cache_option = 'check', cache_file = 'stuff')
#
#Here "stuff.pkl" is name of the file from which loading or saving happens.
#The program will check whether stuff.pkl exists.
#  If it does, it will load stuff.pkl and return that.
#  If it doesn't, it will run the function, save it to stuff.pkl, and return the result.
#If option is not given, then default_option is used.
#If file is not given then e.g. 'calculator' would be used in above example.
def cacheable(file_type = 'pkl', default_option = 'check', default_file = None, inherit_prefix = False):
  def load(path):
    return cache_load(path)

  def run(func, *args, **kwargs):
    result = func(*args, **kwargs)
    return result

  def save(func, path, *args, **kw):
    result = run(func, *args, **kw)
    cache_save(result, path, file_type)
    return result

  def check(func, path, *args, **kw):
    if os.path.isfile(path):
      return load(path)
    else:
      return save(func, path, *args, **kw)
  
  def caching_wrapper(func):
    def wrap(*args, **kw):
      option_arg = 'cache_option' 
      if option_arg in kw:
        option = kw[option_arg]
        kw.pop(option_arg)
      else:
        option = default_option
      
      #Get file name
      path_arg = 'cache_file'
      if path_arg in kw:
        path = kw[path_arg]
        kw.pop(path_arg)
      else:
        path = default_file

      #Check whether to pass prefix to wrapped function.
      if inherit_prefix:
        kw['cache_prefix'] = path
      
      #Get file path
      if option == 'run':
        #File is not necessary for just running a function
        pass
      else:
        if path is None:
          raise Exception('Someone didn\'t define a file to load/save to, for a cacheable function.')
        path = cache_path + path
        if file_type == 'pkl':
          path = path + '.pkl'
        elif file_type == 'json':
          path = path + '.json'
        else:
          raise ValueError('File type ' + file_type + ' not supported.')

      if option == 'load':
        return load(path)
      elif option == 'run':
        return run(func, *args, **kw)
      elif option == 'save':
        return save(func, path, *args, **kw)
      elif option == 'check':
        return check(func, path, *args, **kw)
      else:
        raise ValueError('Caching option ' + option + ' not recognized.')

    return wrap
  return caching_wrapper
  
#Adds a string to cache_prefix/cache_file string.
#Returns updated string.
#If cache_prefix is None returns None.
#This is used to get a heirarchical tree of cache file names.
def update_cache_prefix(cache_prefix, addition):
  if cache_prefix is None:
    return None
  return cache_prefix + '__' + addition
  
#Decorator that prints when function starts and ends working.
#Prints if function is run with parameter progress = True.
#process_name - how to name process in messages.
#terminal - if True doesn't pass progress as parameter to decorated function.
def progress(process_name, terminal = False):
  def progress_wrapper(func):
    def wrap(*args, progress = None, **kw):
      if progress is None:
        progress = False

      if progress:
        print('Starting ' + process_name)
      
      def run():
        if terminal or 'progress' in kw:
          return func(*args, **kw)
        else:
          return func(*args, progress = progress, **kw)
      
      if progress:
        print('Finished ' + process_name)
      else:
        result = run()
      
      return result
    return wrap
  return progress_wrapper

def batcheable(data_keys = [], default_batch_size = 256, default_value = [], 
               combine_func = None, update_prefix = False):
  #Set up default function for combination of batch outputs.
  if combine_func is None:
    combine_func = lambda x, y: x + y
  
  key_count = len(data_keys)
  if key_count == 0:
    raise ValueError('No data keys to batch passed to batcheable decorator.')
  def batching_wrapper(func):
    def process_batches(*args, cache_prefix = None, batch_size = None,
                        batch_args = {}, **kw):
      if 'progress' in kw:
        progress = kw['progress']
      else:
        progress = False

      if batch_size is None:
        batch_size = default_batch_size
      
      #Get data
      data = []
      data = list(args[:key_count])
      if len(args) < key_count:
        for key in data_keys[len(args):]:
          if key in kw:
            data.append(kw[key])
            kw.pop(key)
          else:
            raise ValueError('Batchable function didn\'t receive all necessary data arguments.')

      size = -1
      for data_list in data:
        if size == -1:
          size = len(data_list)
        elif size != len(data_list):
          raise ValueError('Lists of data passed to batching function are not of equal length.')

      #Run on batches
      result = default_value
      for offset in range(0, size, batch_size):
        if progress:
          print(offset, '/', size)

        data_slice = []
        for data_list in data:
          data_slice.append(data_list[offset:offset + batch_size])

        batch_range = (offset, min(offset + batch_size, size))

        #Add batch range to cache_file.
        extra_args = {}
        if update_prefix\
           and 'cache_file' not in batch_args:
          cache_file = update_cache_prefix_batch(cache_prefix, batch_range)
          extra_args['cache_file'] = cache_file
        
        func_output = func(*data_slice, **batch_args, **extra_args, **kw)
        
        if offset == 0:
          result = func_output
        else:
          result = combine_func(result, func_output)
      
      return result

    return process_batches

  return batching_wrapper

def range_tostr(range):
  return str(range[0]) + '-' + str(range[1])

def batch_range_tostr(batch_range):
  return 'batch_' + range_tostr(batch_range)

#Adds a string representing the batch range to cache_prefix/cache_file string.
#Returns updates string.
def update_cache_prefix_batch(cache_prefix, batch_range):
  addition = batch_range_tostr(batch_range)
  return update_cache_prefix(cache_prefix, addition)

def get_CLIP_similarity_matrix_low(text_encodings, image_encodings):
  #image_encodings = image_encodings / np.linalg.norm(image_encodings, axis=1, keepdims = True)
  #text_encodings = text_encodings / np.linalg.norm(text_encodings, axis=1, keepdims = True)
  result = text_encodings @ image_encodings.T
  return result

#Determininstically shuffles list according to seed. 
def deterministic_shuffle(ls, seed = 42):
  import random
  random.seed(seed)

  #Generate permutation
  if len(ls) == 0:
    return []
  perm = [0]
  for idx in range(1, len(ls)):
    perm.append(idx)
    tmp_idx = random.randint(0, idx)
    tmp = perm[tmp_idx]
    perm[tmp_idx] = perm[idx]
    perm[idx] = tmp
  
  #Permute ls
  result = []
  for idx in perm:
    result.append(ls[idx])
  
  return result

def deterministic_shuffle_np(array, seed = 42):
  import numpy as np
  perm = deterministic_shuffle(range(array.shape[0]), seed)
  res = []
  for idx in perm:
    res.append(array[idx])
  res = np.array(res)
  return res

img_encodings = cache_load('./cache/2020_image_clip_encodings.pkl')
img_encodings = deterministic_shuffle_np(img_encodings)[:100]

#Takes one text.
#Returns array of clip similarities for the 100 images.
def get_clip_sim(text):
    eng_text = translate([text])
    text_enc = clip_encode_text(eng_text)
    return get_CLIP_similarity_matrix_low(text_enc, img_encodings)
    
#Function that installs and imports all neccessary stuff for 
#Flair Name Entity Recognition.
ner_configed = False
def config_ner():
  global ner_configed
  if not ner_configed:
    from flair.data import Sentence
    global ner_tagger
    from flair.models import SequenceTagger
    ner_tagger = SequenceTagger.load('ner')
    ner_configed = True

letters = ['a', 'ā', 'b', 'c', 'č', 'd', 'e', 'ē', 'f', 'g', 'ģ', 'h', 'i', 'ī', 'j', 'k', 'ķ', 'l', 'ļ', 'm', 'n', 'ņ', 'o', 'p', 'q', 'r', 's', 'š', 't', 'u', 'ū', 'v', 'w', 'x', 'y', 'z', 'ž']
#Symbols that can be contained in namewords
name_symbols = letters + [char.upper() for char in letters] + ['\'']

#Function that splits text up into words.
#Also returns list of boolean flags for whether a word is followed by punctuation.
#  This is useful, because full names don't usually contain punctuation.
def words_and_punctuation(text):
  words = []
  punctuation = []

  #Check if word is nonempty, and add to list of words
  def check_add(word):
    if len(word) > 0:
      words.append(word)
      punctuation.append(False)

  word = ''
  for char in text:
    if char in name_symbols:
      word+= char
    else:
      check_add(word)
      word = ''
      if char != ' ' and char != '-' and len(punctuation) > 0:
        punctuation[-1] = True
  check_add(word)

  return (words, punctuation)

@progress('name extraction from texts using Falir NER')
@cacheable(file_type = 'json', default_option = 'run')
def extract_text_fullnames_ner(texts, progress = False, **kw):
  config_ner()
  from flair.data import Sentence
  global ner_tagger

  result = []
  for idx, text in enumerate(texts):
    if progress and idx % 1024 == 0:
      print(idx, '/', len(texts))
    names = []
    sentence = Sentence(text)
    ner_tagger.predict(sentence)
    for span in sentence.get_spans('ner'):
      if span.tag == 'PER':
        words, punct = words_and_punctuation(span.text)
        if len(words) >= 2:
          names.append(words)
    result.append(names)

  return result

#Finds names in texts.
#Takes list of texts.
#Returns list of lists of human names found in texts.
#method:
#  exact_match - considers a word a name-word if found exactly in dictionary of names.
#  ner - uses Flair Name Entity Recognition neural net.
@progress('name extraction from texts')
@cacheable(file_type = 'json', inherit_prefix = True, default_option = 'run')
def extract_text_fullnames(texts, method = 'ner', method_args = {},
                           cache_prefix = None, **kw):
  if method == 'exact_match':
    allowed_names = get_namewords(**kw)
    if 'cache_file' not in method_args:
      cache_file = update_cache_prefix(cache_prefix, 'exact_match')
      method_args['cache_file'] = cache_file
    return extract_text_fullnames_exact_match(texts, allowed_names, 
                                              **method_args, **kw)
  if method == 'ner':
    if 'cache_file' not in method_args:
      cache_file = update_cache_prefix(cache_prefix, 'ner')
      method_args['cache_file'] = cache_file
    return extract_text_fullnames_ner(texts, **method_args, **kw)
  else:
    raise ValueError('Name extraction method ' + name_extraction_method + ' not recognized.')

def separate_clusters(clusters):
  result_names = []
  result_encodings = []
  for cluster in clusters:
    result_names.append(cluster[0])
    result_encodings.append(cluster[1])
  return (result_names, result_encodings)

clusts = cache_load('cache/test_clusters_2019_ner.pkl')
clusts = clusts[0]
name_clusts, face_clusts = separate_clusters(clusts)

#Returns number from 0 to 1, which represents how different strings are.
#1 - completely different.
#0 - completely identical.
#Uses edit distance.
def CER_string_metric(word1, word2):
  import editdistance
  return editdistance.distance(word1, word2) / max(len(word1), len(word2))

def cat_strings(strings):
  res = ''
  for idx, string in enumerate(strings):
    if idx!= 0:
      res+= ' '
    res+= string
  return res

#Uses string_metric on contatenated names. 
def CER_name_metric(name1, name2):
  return CER_string_metric(cat_strings(name1), cat_strings(name2))

import numpy as np

#Find which mugshot clusters match with name.
#If none match well enough returns empty set.
#name - fullname that we're matching.
#cluster_names - list of cluster fullname examples.
#  i.e. list of list of fullnames.
def match_fullname_to_clusters(name, cluster_names, metric = 'char_error_rate',
                               method = 'closest', cutoff = 0.2, **kw):
  if metric == 'char_error_rate':
    if method == 'closest':
      indexes = []
      names = []
      for cluster_idx, name_list in enumerate(cluster_names):
        indexes+= [cluster_idx for i in range(len(name_list))]
        names+= name_list
      distances = [CER_name_metric(name, name2) for name2 in names]
      min_idx = np.argmin(distances)
      if distances[min_idx] > cutoff:
        return []
      else:
        return [indexes[min_idx]]
    else:
      raise ValueError('Fullname cluster matching method ' + method + ' not recognized.')
  else:
    raise ValueError('Fullname cluster matching metric ' + metric + ' not recognized.')


#Find which mugshot clusters match with list of names.
#Takes list of fullnames, and list cluster fullname examples.
#Returns list of lists of matches.
def match_fullname_list_to_clusters(names, *args, **kw):
  result = []
  for name in names:
    result.append(match_fullname_to_clusters(name, *args, **kw))
  return result

@progress('matching names to name clusters', terminal = True)
@cacheable(file_type = 'json', default_option = 'run')
def match_fullnames_to_clusters(names, *args, **kw):
  result = []
  for name_list in names:
    matches = match_fullname_list_to_clusters(name_list, *args, **kw)
    tmp = []
    for match_list in matches:
      tmp+= match_list
    result.append(tmp)
  return result

#Finds name cluster indexes that match names in text
def name_clusts_found(text):
    fullnames = extract_text_fullnames([text])
    return match_fullnames_to_clusters(fullnames, name_clusts)

probs = cache_load('cache/test_cluster_probs_2019_ner.json')
image_faces = cache_load('cache/test_2019_ner_test_iclust.json')[:100]

import math
    
#Function that returns similarity matrix between mugshot clusters found in images
#and mugshot clusters found in descriptions.
#Takes list of lists of mugshot clusters found in descriptions and images.
#Returns matrix of evidence values.
#d_clusters - list of lists of mugshot cluster indexes found in descriptions.
#i_clusters - list of lists of mugshot cluster indexes found in images.
#probabilities - object returned by "get_mugshot_cluster_probabilities"
@progress('calculation of face similarity matrix between clusters')
@cacheable(file_type = 'pkl', default_option = 'run')
def get_LR_cluster_face_similarity_low(d_clusters, i_clusters, probabilities, progress = False):
  result = np.zeros(shape = [len(d_clusters), len(i_clusters)], dtype = np.float32)

  for d_idx, d_cluster_list in enumerate(d_clusters):
    for i_idx, i_cluster_list in enumerate(i_clusters):
      evidence = 0
      union = list(set().union(d_cluster_list, i_cluster_list))
      for clust_idx in union:
        p = probabilities[clust_idx]
        evidence-= math.log(p['nI|nD'] / p['nI'])
        if clust_idx in d_cluster_list:
          if clust_idx in i_cluster_list:
            evidence+= math.log(p['I|D'] / p['I'])
          else:
            evidence+= math.log(p['nI|D'] / p['nI'])
        else:
          if clust_idx in i_cluster_list:
            evidence+= math.log(p['I|nD'] / p['I'])
          else:
            evidence+= math.log(p['nI|nD'] / p['nI'])
      result[d_idx, i_idx] = evidence
  
  return result

#Returns array of similarities based on face sim.
def get_face_sim(text):
    text = translate([text])
    fullnames = extract_text_fullnames(text)
    clusters = match_fullnames_to_clusters(fullnames, name_clusts)
    return get_LR_cluster_face_similarity_low(clusters, image_faces, probs)
    
def get_general_sim(text):
    f = get_face_sim(text)
    c = get_clip_sim(text)
    return f + c
    

def get_top_k(text):
    sim = get_general_sim(text)
    val, idx = torch.topk(torch.tensor(sim), k = 5)

    return idx[0].data.numpy()
    
def get_images(text):
    idx = get_top_k(text)
    return idx