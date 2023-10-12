import bz2
import codecs
import datetime
import gzip
import locale
import logging
import re
import sys

from . import get_version

try:
    # In Python2 import cPickle for better performance
    import cPickle as pickle
except ImportError:
    import pickle

PY3 = sys.version_info.major == 3

_logger = logging.getLogger(__name__)


class MorfessorIO:
    """Definition for all input and output files. Also handles all
    encoding issues.

    The only state this class has is the separators used in the data.
    Therefore, the same class instance can be used for initializing multiple
    files.

    """

    def __init__(self, encoding=None, construction_separator=' + ',
                 comment_start='#', compound_separator='\s+',
                 atom_separator=None, lowercase=False):
        self.encoding = encoding
        self.construction_separator = construction_separator
        self.comment_start = comment_start
        self.compound_separator = compound_separator
        self.atom_separator = atom_separator
        if atom_separator is not None:
            self._atom_sep_re = re.compile(atom_separator, re.UNICODE)
        self.lowercase = lowercase

    def read_segmentation_file(self, file_name, has_counts=True, **kwargs):
        """Read segmentation file.

        File format:
        <count> <construction1><sep><construction2><sep>...<constructionN>

        """
        _logger.info("Reading segmentations from '%s'..." % file_name)
        for line in self._read_text_file(file_name):
            if has_counts:
                count, compound = line.split(' ', 1)
            else:
                count, compound = 1, line
            yield int(count), compound.split(self.construction_separator)
        _logger.info("Done.")

    def write_segmentation_file(self, file_name, segmentations, **kwargs):
        """Write segmentation file.

        File format:
        <count> <construction1><sep><construction2><sep>...<constructionN>

        """
        _logger.info("Saving segmentations to '%s'..." % file_name)
        with self._open_text_file_write(file_name) as file_obj:
            d = datetime.datetime.now().replace(microsecond=0)
            file_obj.write("# Output from Morfessor Baseline %s, %s\n" %
                           (get_version(), d))
            for count, segmentation in segmentations:
                if self.atom_separator is None:
                    s = self.construction_separator.join(segmentation)
                else:
                    s = self.construction_separator.join(
                        map(lambda x: ' '.join(x), segmentation))
                file_obj.write("%d %s\n" % (count, s))
        _logger.info("Done.")

    def read_corpus_files(self, file_names):
        """Read one or more corpus files.

        Yield for each compound found (1, compound, compound_atoms).

        """
        for file_name in file_names:
            for item in self.read_corpus_file(file_name):
                yield item

    def read_corpus_list_files(self, file_names):
        """Read one or more corpus list files.

        Yield for each compound found (count, compound, compound_atoms).

        """
        for file_name in file_names:
            for item in self.read_corpus_list_file(file_name):
                yield item

    def read_corpus_file(self, file_name):
        """Read one corpus file.

        For each compound, yield (1, compound, compound_atoms).
        After each line, yield (0, \"\\n\", ()).

        """
        _logger.info("Reading corpus from '%s'..." % file_name)
        compound_sep = re.compile(self.compound_separator, re.UNICODE)
        for line in self._read_text_file(file_name):
            for compound in compound_sep.split(line):
                if len(compound) > 0:
                    yield 1, compound, self._split_atoms(compound)
            yield 0, "\n", ()
        _logger.info("Done.")

    def read_corpus_list_file(self, file_name):
        """Read a corpus list file.

        Each line has the format:
        <count> <compound>

        Yield tuples (count, compound, compound_atoms) for each compound.

        """
        _logger.info("Reading corpus from list '%s'..." % file_name)
        for line in self._read_text_file(file_name):
            try:
                count, compound = line.split(None, 1)
                yield int(count), compound, self._split_atoms(compound)
            except ValueError:
                yield 1, line, self._split_atoms(line)
        _logger.info("Done.")

    def read_annotations_file(self, file_name, construction_separator=' ',
                              analysis_sep=','):
        """Read a annotations file.

        Each line has the format:
        <compound> <constr1> <constr2>... <constrN>, <constr1>...<constrN>, ...

        Yield tuples (compound, list(analyses)).

        """
        annotations = {}
        _logger.info("Reading annotations from '%s'..." % file_name)
        for line in self._read_text_file(file_name):
            analyses = []
            compound, analyses_line = line.split(None, 1)

            if compound not in annotations:
                annotations[compound] = []

            if analysis_sep is not None:
                for analysis in analyses_line.split(analysis_sep):
                    analysis = analysis.strip()
                    annotations[compound].append(
                        analysis.strip().split(construction_separator))
            else:
                annotations[compound].append(
                    analyses_line.split(construction_separator))

        _logger.info("Done.")
        return annotations

    def write_lexicon_file(self, file_name, lexicon):
        """Write to a Lexicon file all constructions and their counts."""
        _logger.info("Saving model lexicon to '%s'..." % file_name)
        with self._open_text_file_write(file_name) as file_obj:
            for construction, count in lexicon:
                file_obj.write("%d %s\n" % (count, construction))
        _logger.info("Done.")

    def read_binary_model_file(self, file_name):
        """Read a pickled model from file."""
        _logger.info("Loading model from '%s'..." % file_name)
        with open(file_name, 'rb') as fobj:
            model = pickle.load(fobj)
        _logger.info("Done.")
        return model

    def write_binary_model_file(self, file_name, model):
        """Pickle a model to a file."""
        _logger.info("Saving model to '%s'..." % file_name)
        with open(file_name, 'wb') as fobj:
            pickle.dump(model, fobj, pickle.HIGHEST_PROTOCOL)
        _logger.info("Done.")

    def read_any_model(self, file_name):
        """Read a file that is either a binary model or a Morfessor 1.0 style
        model segmentation. This method can not be used on standard input as
        data might need to be read multiple times"""
        try:
            model = self.read_binary_model_file(file_name)
            _logger.info("%s was read as a binary model" % file_name)
            return model
        except BaseException:
            pass

        from morfessor import BaselineModel
        model = BaselineModel()
        model.load_segmentations(self.read_segmentation_file(file_name))
        _logger.info("%s was read as a segmentation" % file_name)
        return model

    def _split_atoms(self, construction):
        """Split construction to its atoms."""
        if self.atom_separator is None:
            return construction
        else:
            return tuple(self._atom_sep_re.split(construction))

    def _open_text_file_write(self, file_name):
        """Open a file with the appropriate compression and encoding"""
        if file_name == '-':
            file_obj = sys.stdout
            if PY3:
                return file_obj
        elif file_name.endswith('.gz'):
            file_obj = gzip.open(file_name, 'wb')
        elif file_name.endswith('.bz2'):
            file_obj = bz2.BZ2File(file_name, 'wb')
        else:
            file_obj = open(file_name, 'wb')
        if self.encoding is None:
            # Take encoding from locale if not set so far
            self.encoding = locale.getpreferredencoding()
        return codecs.getwriter(self.encoding)(file_obj)

    def _read_text_file(self, file_name):
        """Read a text file with the appropriate compression and encoding.

        Comments and empty lines are skipped.

        """
        encoding = self.encoding
        if encoding is None:
            if file_name != '-':
                encoding = self._find_encoding(file_name)

        if file_name == '-':

            if PY3:
                inp = sys.stdin
            else:
                class StdinUnicodeReader:
                    def __init__(self, encoding):
                        self.encoding = encoding
                        if self.encoding is None:
                            self.encoding = locale.getpreferredencoding()

                    def __iter__(self):
                        return self

                    def next(self):
                        l = sys.stdin.readline()
                        if not l:
                            raise StopIteration()
                        return l.decode(self.encoding)
                inp = StdinUnicodeReader(encoding)
        else:
            if file_name.endswith('.gz'):
                file_obj = gzip.open(file_name, 'rb')
            elif file_name.endswith('.bz2'):
                file_obj = bz2.BZ2File(file_name, 'rb')
            else:
                file_obj = open(file_name, 'rb')

            if self.encoding is None:
                self.encoding = self._find_encoding(file_name)

            inp = codecs.getreader(self.encoding)(file_obj)

        try:
            for line in inp:
                line = line.rstrip()
                if len(line) > 0 and not line.startswith(self.comment_start):
                    if self.lowercase:
                        yield line.lower()
                    else:
                        yield line
        except KeyboardInterrupt:
            if file_name == '-':
                _logger.info("Finished reading from stdin")
                return
            else:
                raise

    def _find_encoding(self, *files):
        """Test default encodings on reading files.

        If no encoding is given, this method can be used to test which
        of the default encodings would work.

        """
        test_encodings = ['utf-8', locale.getpreferredencoding()]
        for encoding in test_encodings:
            ok = True
            for f in files:
                if f == '-':
                    continue
                try:
                    if f.endswith('.gz'):
                        file_obj = gzip.open(f, 'rb')
                    elif f.endswith('.bz2'):
                        file_obj = bz2.BZ2File(f, 'rb')
                    else:
                        file_obj = open(f, 'rb')

                    for _ in codecs.getreader(encoding)(file_obj):
                        pass
                except UnicodeDecodeError:
                    ok = False
                    break
            if ok:
                _logger.info("Detected %s encoding" % encoding)
                return encoding

        raise UnicodeError("Can not determine encoding of input files")
