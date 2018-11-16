import numpy as np

class WV_opt():
    def __init__(self, lamb, W0):
        self.A = None
        self.lamb = lamb
        self.W0 = W0.T
        self.iteration = 0
        self.alpha = 0

    def fit(self, summary, document):
        """

        :param summary: list of summary vectors list
        :param document: list of document vectors
        :return: A
        """

        distance_vecs = []



        weight = self.W0.dot(self.W0.T)


        for s_vecs, d_vec in zip(summary, document):
            for s_vec in s_vecs:
                distance_vec = d_vec - s_vec
                distance_vecs.append(distance_vec.dot(distance_vec.T))


        for i in range(self.iteration):
            grad = np.zeros_like(self.A)
            for distance_matrix in distance_vecs:
                grad += distance_matrix
            grad = 2 * self.A.dot(grad)

            grad += 2 * self.lamb * self.A.dot(weight)
            grad -= 2 * self.lamb * weight

            self.A = self.A - self.alpha * grad

        print np.linalg.norm(self.A.dot(summary[0][0]) - self.A.dot(document[0]))
        print np.linalg.norm(self.A.dot(summary[0][1]) - self.A.dot(document[0]))
        print np.linalg.norm(self.A.dot(summary[0][2]) - self.A.dot(document[0]))

    def fit_inverse(self, summary, document):
        self.A = np.zeros((self.W0.shape[0], self.W0.shape[0]))
        weight = self.W0.dot(self.W0.T)
        distance_vecs = []

        documents_num = len(document)

        #print summary
        #print document

        for s_vecs, d_vec in zip(summary, document):
            for s_vec in s_vecs:
                distance_vec = d_vec - s_vec
                distance_vecs.append((distance_vec.dot(distance_vec.T)) / len(s_vecs) / documents_num)

        for distance_matrix in distance_vecs:
            self.A += distance_matrix

        self.A += self.lamb * weight
        self.A = np.linalg.inv(self.A)
        self.A = self.lamb * weight.dot(self.A)

        #print np.linalg.norm(self.A.dot(summary[0][0]) - self.A.dot(document[0]))
        #print np.linalg.norm(self.A.dot(summary[0][1]) - self.A.dot(document[0]))
        #print np.linalg.norm(self.A.dot(summary[0][2]) - self.A.dot(document[0]))

        return self.A


if __name__ == '__main__':
    W0 = np.array([[1,0],
                  [2,2],
                  [3,3]])

    d = np.c_[np.array([4,3])]
    s = np.c_[np.array([2,2])]
    s2 = np.c_[np.array([4,4])]
    s3 = np.c_[np.array([6,2])]

    print np.linalg.norm(d- s)
    print np.linalg.norm(d- s2)
    print np.linalg.norm(d- s3)


    wv = WV_opt(100, 0.01, 0.001, W0)
    wv.fit_inverse([[s, s2, s3]], [d])








