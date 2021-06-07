from typing import Sequence, Any
import numpy as np

class Index:

    def __init__(self, vocab: Sequence[Any], start=0):
        """
        Assigns an index to each unique item in the `vocab` iterable,
        with indexes starting from `start`.

        Indexes should be assigned in order, so that the first unique item in
        `vocab` has the index `start`, the second unique item has the index
        `start + 1`, etc.
        """
        self.vocab = vocab
        self.start = start

        unique_vocab = {}
        unique_index = {}

        for value in vocab:
            if value not in unique_index:
                unique_index[value] = start
                unique_vocab[start] = value
                # print(unique_vocab)
                start += 1

        self.unique_vocab = unique_vocab
        self.unique_index = unique_index


    def objects_to_indexes(self, object_seq: Sequence[Any]) -> np.ndarray:   # -> np.ndarray indicates the type that the function returns
        """
        Returns a vector of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.

        :param object_seq: A sequence of objects.
        :return: A 1-dimensional array of the object indexes.
        """

        indexes = []

        for value in object_seq:
            if value not in self.unique_index:
                indexes.append(self.start - 1)
            else:
                indexes.append(self.unique_index[value])

        return np.array(indexes)

    def objects_to_index_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a matrix of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.

        If the sequences are not all of the same length, shorter sequences will
        have padding added at the end, with `start-1` used as the pad value.

        :param object_seq_seq: A sequence of sequences of objects.
        :return: A 2-dimensional array of the object indexes.
        """
        index_matrix = []

        # To find the width of the 2D array
        width = len(max(object_seq_seq, key=len))

        # Iterate through the given objects and get the indexes saved in the index_matrix
        for i in object_seq_seq:
            index_matrix_temp = []
            for j in i:
                if j in self.unique_index:
                    index_matrix_temp.append(self.unique_index[j])
                else:
                    index_matrix_temp.append(self.start - 1)
            # Padding
            for k in range(len(index_matrix_temp), width):
                index_matrix_temp.append(self.start - 1)
            index_matrix.append(np.array(index_matrix_temp))

        return np.array(index_matrix)

    def objects_to_binary_vector(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a binary vector, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq: A sequence of objects.
        :return: A 1-dimensional array, with 1s at the indexes of each object,
                 and 0s at all other indexes.
        """

        objects_dict = {i: object_seq[i] for i in range(0, len(object_seq))}
        binary_vector = np.zeros((len(self.vocab) + self.start), dtype=int)

        for word in self.unique_vocab.values() and objects_dict.values():
            binary_vector[self.unique_index.get(word)] = 1

        return binary_vector

    def objects_to_binary_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a binary matrix, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq_seq: A sequence of sequences of objects.
        :return: A 2-dimensional array, where each row in the array corresponds
                 to a row in the input, with 1s at the indexes of each object,
                 and 0s at all other indexes.
        """

        binary_matrix = []

        for i in object_seq_seq:
            binary_matrix.append(self.objects_to_binary_vector(i))

        binary_matrix = np.array(binary_matrix)

        return binary_matrix


    def indexes_to_objects(self, index_vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of objects associated with the indexes in the input
        vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_vector: A 1-dimensional array of indexes
        :return: A sequence of objects, one for each index.
        """

        indexes_to_objects = []

        for i in index_vector:
            if i in self.unique_vocab:
                indexes_to_objects.append(self.unique_vocab[i])

        return indexes_to_objects

    def index_matrix_to_objects(
            self, index_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects associated with the indexes
        in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_matrix: A 2-dimensional array of indexes
        :return: A sequence of sequences of objects, one for each index.
        """

        index_matrix_to_objects = []

        for i in index_matrix:
            index_matrix_to_objects.append(self.indexes_to_objects(i))

        return index_matrix_to_objects

    def binary_vector_to_objects(self, vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of the objects identified by the nonzero indexes in
        the input vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param vector: A 1-dimensional binary array
        :return: A sequence of objects, one for each nonzero index.
        """

        objects = []
        index = 0

        for i in vector:
            if i != 0:
                objects.append(self.unique_vocab[index])
            index += 1

        return objects

    def binary_matrix_to_objects(
            self, binary_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects identified by the nonzero
        indices in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param binary_matrix: A 2-dimensional binary array
        :return: A sequence of sequences of objects, one for each nonzero index.
        """

        matrix_objects = []
        for i in binary_matrix:
            matrix_objects.append(self.binary_vector_to_objects(i))

        return matrix_objects

