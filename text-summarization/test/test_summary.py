
import os
import subprocess
from tensorflow.core.example import example_pb2
import struct

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

def tokenize_news(news_dir, tokenized_news_dir):
#     for more than one news:
#     """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
    print("Preparing to tokenize %s to %s..." % (news_dir, tokenized_news_dir))
    news = os.listdir(news_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for s in news:
          f.write("%s \t %s\n" % (os.path.join(news_dir, s), os.path.join(tokenized_news_dir, s)))

    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing %i files in %s and saving in %s..." % (len(news), news_dir, tokenized_news_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")

    # Check that the tokenized news directory contains the same number of files as the original directory
    num_orig = len(os.listdir(news_dir))
    num_tokenized = len(os.listdir(tokenized_news_dir))
    if num_orig != num_tokenized:
        raise Exception("The tokenized news directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_news_dir, num_tokenized, news_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (news_dir, tokenized_news_dir))

def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines

def get_art_abs(story_file):
	lines = read_text_file(story_file)

	# Lowercase everything
	lines = [line.lower() for line in lines]

	# # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
	# lines = [fix_missing_period(line) for line in lines]

	# Separate out article and abstract sentences
	article_lines = []
	highlights = []
	next_is_highlight = False
	for idx,line in enumerate(lines):
		if line == "":
		  continue # empty line
		elif line.startswith("@highlight"):
		  next_is_highlight = True
		elif next_is_highlight:
		  highlights.append(line)
		else:
		  article_lines.append(line)

	# Make article into a single string
	article = ' '.join(article_lines)

	# Make abstract into a signle string, putting <s> and </s> tags around the sentences
	abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

	return article, abstract



news_dir = '/Users/sze/Desktop/nlp/text-summarization/test/news'
tokenized_news_dir = '/Users/sze/Desktop/nlp/text-summarization/test/pre-tokenized'
finished_files_dir = '/Users/sze/Desktop/nlp/text-summarization/test/tokenized_news'
story_file = '/Users/sze/Desktop/nlp/text-summarization/test/pre-tokenized/test01.story'

if not os.path.exists(tokenized_news_dir): 
    os.makedirs(tokenized_news_dir)
if not os.path.exists(finished_files_dir): 
    os.makedirs(finished_files_dir)   


 # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
tokenize_news(news_dir, tokenized_news_dir)

# some preprocessing of toeknized text
article, abstract = get_art_abs(story_file)

# at this stage, only need article 
with open(os.path.join(finished_files_dir,"tokenized_news.bin"), "wb") as writer:
    # Write to tf.Example
    tf_example = example_pb2.Example()
    tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
    tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))

    print("Finished writing file tokenized_news.bin")

