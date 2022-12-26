import collections
import unicodedata

def convert_to_unicode(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))

def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding='utf8') as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

def convert_by_vocab(vocab, items):
  output = []
  for item in items:
    output.append(vocab[item])
  return output

def convert_tokens_to_ids(vocab, tokens):
  return convert_by_vocab(vocab, tokens)

def whitespace_tokenize(text):
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens


class FullTokenizer(object):
    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.max_input_chars_per_word = 200

    def tokenize(self, text):
        all_tokens = []
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        for token in output_tokens:
            tok = convert_to_unicode(token)
            final_tokens = []
            for t in whitespace_tokenize(tok):
                chars = list(t)
                if len(chars) > self.max_input_chars_per_word:
                    final_tokens.append("[UNK]")
                    continue
                is_bad = False
                start = 0
                sub_tokens = []
                while start < len(chars):
                    end = len(chars)
                    cur_substr = None
                    while start < end:
                        substr = "".join(chars[start:end])
                        if start > 0:
                            substr = "##" + substr
                        if substr in self.vocab:
                            cur_substr = substr
                            break
                        end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if is_bad:
                final_tokens.append("[UNK]")
            else:
                final_tokens.extend(sub_tokens)
            for sub_token in final_tokens:
                all_tokens.append(sub_token)
        return all_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or unicodedata.category(char).startswith("C"): # char == "\t" or char == "\n" or char == "\r" -> return false
                continue
            if char == " " or char == "\t" or char == "\n" or char == "\r" or unicodedata.category(char) == "Zs": # is white space or not
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _run_strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            cp = ord(char)
            if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or 
                (cp >= 123 and cp <= 126)) or unicodedata.category(char).startswith("P"): # is pintuation?
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if ((cp >= 0x4E00 and cp <= 0x9FFF) or (cp >= 0x3400 and cp <= 0x4DBF) or (cp >= 0x20000 and cp <= 0x2A6DF) or 
                (cp >= 0x2A700 and cp <= 0x2B73F) or (cp >= 0x2B740 and cp <= 0x2B81F) or (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or (cp >= 0x2F800 and cp <= 0x2FA1F)):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)