import numpy as np
from typing import Optional, Tuple, Sequence
from functools import cached_property

# from pqsim.randoms import RandomsFactory as PyRandomsFactory


class RandomVariable:
    def eval(self) -> float:
        raise NotImplementedError


class MarkedProcessVariable:
    def eval(self) -> Tuple[int, float]:
        raise NotImplementedError


class MmapVariable(MarkedProcessVariable):
    def __init__(self, d: Sequence[np.ndarray], p0: np.ndarray):
        """
        Constructor of MMAP variable.

        Since computation of initial PMF may be hard in other implementations
        (e.g., in C++) and to unify variable interface, we expect that
        initial PMF is computed outside, probably - in Mmap constructor.

        Moreover, no D or p0 validation is performed here. It is expected
        that data was validated before random variable creation.

        Parameters
        ----------
        d : sequence of np.ndarray
            matrices of MMAP
        p0 : np.ndarray
            initial probability distribution
        """
        # 1) Compute rates and order
        rates = -d[0].diagonal()
        self._order = len(rates)

        # 2) Then, we need to store cumulative transition probabilities P.
        #    P - matrix of shape N x (K*N), where P[I, J] is a prob., that:
        #      - new state will be J (mod N), and
        #      - if J < N, then no packet is generated, or
        #      - if J >= N, then packet of type J // N is generated.
        self._trans_pmf = np.hstack((
            d[0] + np.diag(rates),  # leftmost diagonal is zero
            *d[1:]
        )) / rates[:, None]

        # 3) Define random variables generators:
        # - random generators for time in each state:
        self.__rate_rnd = [Rnd(
            lambda n, r=r: np.random.exponential(1/r, size=n))
            for r in rates
        ]

        # - random generators of state transitions:
        trans_range = np.arange(self._order * len(d))
        self.__trans_rnd = [
            Rnd(lambda n, p0=p: np.random.choice(trans_range, p=p0, size=n))
            for p in self._trans_pmf
        ]

        # 4) Since we have the initial PMF p0, we find the initial state:
        self._state = np.random.choice(np.arange(self._order), p=p0)

    def eval(self) -> Tuple[int, float]:
        """
        Get next random value.
        """
        pkt_type = 0
        interval = 0.0
        # print('> start in state ', self._state)
        i = self._state
        while pkt_type == 0:
            interval += self.__rate_rnd[i]()
            j = self.__trans_rnd[i]()
            pkt_type, i = divmod(j, self._order)
        self._state = i
        return interval, pkt_type


class PhaseTypeVariable(RandomVariable):
    def __init__(self, s: np.ndarray, p: np.ndarray):
        """
        Constructor of PH random variable.

        Parameters
        ----------
        s : np.ndarray
            PH generator (subinfinitesimal matrix)
        p : np.ndarray
            PH initial PMF
        """
        rates = -s.diagonal()
        self._order = len(p)

        self._trans_pmf = np.hstack((
            s + np.diag(rates),
            -s.sum(axis=1)[:, None]
        )) / rates[:, None]

        # Create rnd caches:
        # ------------------
        all_states = np.arange(self._order + 1)
        # - random generator for initial state:
        self.__init_rnd = Rnd(
            lambda n: np.random.choice(all_states[:-1], p=p, size=n))
        # - random generators for time in each state:
        self.__rate_rnd = [Rnd(
            lambda n, r=r: np.random.exponential(1/r, size=n))
            for r in rates
        ]
        # - random generators of state transitions:
        self.__trans_rnd = [Rnd(
            lambda n, p0=p: np.random.choice(all_states, p=p0, size=n)
        ) for p in self._trans_pmf]

    def eval(self) -> float:
        """Get next random value."""
        interval = 0.0
        i = self.__init_rnd()
        while i < self._order:
            interval += self.__rate_rnd[i]()
            i = self.__trans_rnd[i]()
        return interval


class Rnd:
    def __init__(self, fn, cache_size=10000):
        self.__fn = fn
        self.__cache_size = cache_size
        self.__samples = []
        self.__index = cache_size

    def __call__(self):
        if self.__index >= self.__cache_size:
            self.__samples = self.__fn(self.__cache_size)
            self.__index = 0
        x = self.__samples[self.__index]
        self.__index += 1
        return x


class RandomsFactory:
    def createMarkedMap(
        self,
        d: Sequence[np.ndarray],
        p0: np.ndarray
    ) -> MarkedProcessVariable:
        """
        Build a random variable for MMAP process.

        Parameters
        ----------
        d : sequence of np.ndarray
            matrices of MMAP
        p0 : np.ndarray
            initial probability distribution

        Returns
        -------
        var : MmapVariable
        """
        return MmapVariable(d, p0)

    def createPhaseType(
        self,
        s: np.ndarray,
        p: np.ndarray
    ) -> RandomVariable:
        """
        Build a PH random variable.

        Parameters
        ----------
        s : np.ndarray
            PH generator (subinfinitesimal matrix)
        p : np.ndarray
            PH initial PMF

        Returns
        -------
        var : PhaseTypeVariable
        """
        return PhaseTypeVariable(s, p)


default_randoms_factory = RandomsFactory()


class Mmap:
    def __init__(self, *d, factory=None):
        # Validate MMAP:
        # - at least two matrices are provided
        # - all matrices have square shape
        # - all matrices have the same shape
        # - D0 is a subinfinitesimal matrix
        # - Di elements are non-negative for all i >= 1
        # - sum of matrices is an infinitesimal generator
        if len(d) <= 1:
            raise ValueError('At least D0 and D1 must be provided')

        # Before further validations, convert and store matrices and order:
        self._d = [np.asarray(di) for di in d]
        self._order = order_of(self._d[0])

        if not all(is_square(di) for di in self._d):
            raise ValueError('D0 and D1 must be square matrices')
        if not all(order_of(di) == self._order for di in self._d):
            raise ValueError("All matrices must have the same order")
        if not is_subinfinitesimal(self._d[0]):
            raise ValueError("D0 must be subinfinitesimal")
        if any((di < 0).any() for di in self._d[1:]):
            raise ValueError("All Di elements must be non-negative for i >= 1")

        # Store infinitesimal generator:
        self._g = sum(self._d)

        if not is_infinitesimal(self._g):
            raise ValueError("Sum of Di must be infinitesimal")

        # Compute rates and initial PMF
        self._rates = -self._d[0].diagonal()
        a = np.vstack((
            self._g.transpose()[:-1],  # remove one row as redundant
            np.ones(self._order)       # add normalization
        ))
        b = np.asarray([0] * (self._order - 1) + [1])
        self._state_pmf = np.linalg.solve(a, b)

        # Store randoms factory
        self.__factory = factory if factory is not None else \
            default_randoms_factory

    @staticmethod
    def exponential(rate, p=None, nc=1) -> 'Mmap':
        if p is None:
            p = [1/nc] * nc
        else:
            nc = len(p)

        d0 = [[-rate]]
        di = [[[rate * p[i]]] for i in range(nc)]
        return Mmap(d0, *di)

    @staticmethod
    def erlang(rate, scale, p=None, nc=1) -> 'Mmap':
        if p is None:
            p = [1/nc] * nc
        else:
            nc = len(p)

        d0 = np.diag([-rate] * scale) + np.diag([rate] * scale, 1)[:-1, :-1]
        ds = []
        for i in range(nc):
            di = np.zeros((scale, scale))
            di[scale-1, 0] = p[i] * rate
            ds.append(di)
        return Mmap(d0, *ds)

    def d(self, i):
        return self._d[i]

    @property
    def generator(self):
        return self._g

    @property
    def trans_pmf(self):
        return self._trans_pmf

    @property
    def state_pmf(self):
        return self._state_pmf

    @property
    def rates(self):
        return self._rates

    @property
    def order(self):
        return self._order

    @property
    def num_types(self):
        return len(self._d) - 1

    @cached_property
    def rnd(self):
        return self.__factory.createMarkedMap(self._d, self._state_pmf)

    def __call__(self) -> Tuple[int, float]:
        return self.rnd.eval()


class Ph:
    def __init__(self, s, p, factory=None):
        # Convert:
        self._s = np.asarray(s)
        self._init_pmf = np.asarray(p).flatten()

        # Validate data:
        # -------------
        if not is_subinfinitesimal(self._s):
            raise ValueError("matrix S must be sub-infinitesimal")
        if (self._init_pmf < 0).any() or sum(self._init_pmf) != 1.0:
            raise ValueError("initial PMF must be stochastic")
        if order_of(self._init_pmf) != order_of(self._s):
            raise ValueError("initial PMF must have the same order as S")

        # Build internal representations for transitions PMFs and rates:
        # --------------------------------------------------------------
        self._order = order_of(self._init_pmf)
        self._rates = -self._s.diagonal()

        # Store randoms factory
        self.__factory = factory if factory is not None else \
            default_randoms_factory

    @staticmethod
    def exponential(rate) -> 'Ph':
        s = [[-rate]]
        p = [1.0]
        return Ph(s, p)

    @staticmethod
    def erlang(rate, scale) -> 'Ph':
        s = np.diag([-rate] * scale) + np.diag([rate] * scale, 1)[:-1, :-1]
        p = [1.0] + [0.0] * (scale - 1)
        return Ph(s, p)

    @property
    def subgenerator(self) -> np.ndarray:
        return self._s

    @property
    def s(self) -> np.ndarray:
        return self._s

    @property
    def init_pmf(self) -> np.ndarray:
        return self._init_pmf

    @property
    def order(self) -> int:
        return self._order

    @property
    def rates(self) -> np.ndarray:
        return self._rates

    @cached_property
    def rnd(self):
        return self.__factory.createPhaseType(self.s, self._init_pmf)

    def __call__(self) -> float:
        return self.rnd.eval()


def str_array(array: np.ndarray):
    if len(array.shape) == 1:
        # Array is a 1-D vector
        return ", ".join(str(x) for x in array)
    if len(array.shape) > 2:
        raise ValueError('only 1-D and 2-D arrays are supported')
    # Array is 2-D
    return "; ".join(
        ", ".join(str(x) for x in array[i])
        for i in range(array.shape[0]))


def parse_array(string: str, ndim: Optional[int] = None):
    data = []
    for row in string.split(';'):
        cols = [s for s in row.split(',') if s]
        data.append([float(x) for x in cols])

    if ndim is None:
        is_vector = len(data) <= 1 or len(data) > 1 and len(data[0]) == 1
        ndim = 1 if is_vector else 2

    array = np.asarray(data)

    if len(array.shape) == 2 and ndim == 1:
        return array.reshape((array.shape[0] * array.shape[1],))
    elif len(array.shape) == 1 and ndim == 2:
        return array.reshape((1, array.shape[0]))

    return array


def is_square(matrix):
    """Checks that a given matrix has square shape.

    Args:
        matrix (numpy.ndarray, list or tuple): a matrix to test

    Returns:
        bool: True if the matrix as a square shape
    """
    matrix = np.asarray(matrix)
    return len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]


def is_vector(vector):
    """Checks that a given matrix represents a vector. One-dimensional
    lists, tuples or numpy arrays are vectors as well as two-dimensional,
    where one dimension equals 1 (e.g. row-vector or column-vector).

    Args:
        vector: a matrix to be tested

    Returns:
        True if the matrix represents a vector
    """
    vector = np.asarray(vector)
    return len(vector.shape) == 1 or (len(vector.shape) == 2 and (
            vector.shape[0] == 1 or vector.shape[1] == 1))


def order_of(matrix):
    """Returns an order of a square matrix or a vector - a number of rows
    (eq. columns) for square matrix and number of elements for a vector.

    Args:
        matrix: a square matrix or a vector

    Returns:
        A number of rows (eq. columns) for a square matrix;
        a number of elements for a vector
    """
    matrix = np.asarray(matrix)
    if is_square(matrix):
        return matrix.shape[0]
    elif is_vector(matrix):
        return len(matrix.flatten())
    else:
        raise ValueError("square matrix or vector expected")


def is_stochastic(matrix, rtol=1e-05, atol=1e-08):
    """Function checks whether a given matrix is stochastic, i.e. a square
    matrix of non-negative elements whose sum per each row is 1.0.

    All comparisons are performed "close-to", np.allclose() method
    is used to compare values.

    Args:
        matrix: a matrix to be tested
        rtol: relative tolerance, see numpy.allclose for reference
        atol: absolute tolerance, see numpy.allclose for reference

    Returns:
        True if the matrix is stochastic, False otherwise
    """
    matrix = np.asarray(matrix)
    return (is_square(matrix) and
            (matrix >= -atol).all() and
            (matrix <= 1.0 + atol).all() and
            np.allclose(matrix.sum(axis=1), np.ones(order_of(matrix)),
                        rtol=rtol, atol=atol))


def is_infinitesimal(matrix, rtol=1e-05, atol=1e-08):
    """Function checks whether a given matrix is infinitesimal, i.e. a square
    matrix of non-negative non-diagonal elements, sum per each row is zero.

    All comparisons are performed in "close-to" fashion, np.allclose() method
    is used to compare values.

    Args:
        matrix: a square matrix to test
        rtol: relative tolerance, see numpy.allclose for reference
        atol: absolute tolerance, see numpy.allclose for reference

    Returns:
        True if the matrix is infinitesimal, False otherwise
    """
    mat = np.asarray(matrix)
    return (is_square(mat) and
            ((mat - np.diag(mat.diagonal().flatten())) >= -atol).all() and
            np.allclose(np.zeros(order_of(mat)), mat.sum(axis=1),
                        atol=atol, rtol=rtol))


def is_subinfinitesimal(matrix, atol=1e-08):
    """Function checks whether a given matrix is sub-infinitesimal,
    i.e. a square matrix of non-negative non-diagonal elements,
    sum per each row is less or equal to zero and at least one sum is strictly
    less than zero.

    Args:
        matrix: a square matrix to test
        atol: absolute tolerance

    Returns:
        True if the matrix is sub-infinitesimal, False otherwise
    """
    mat = np.asarray(matrix)
    rs = mat.sum(axis=1)
    # noinspection PyUnresolvedReferences
    return (is_square(mat) and
            ((mat - np.diag(mat.diagonal().flatten())) >= -atol).all() and
            (rs <= atol).all() and (rs < -atol).any())


def estimate_mmap_rates(mmap: Mmap, n_iters: int = 1000000):
    """
    Estimate packet arrival rates in MMAP process.
    """
    samples = [mmap() for _ in range(n_iters)]
    timestamps = np.asarray([sample[0] for sample in samples]).cumsum()
    marks = np.asarray([sample[1] for sample in samples])

    # Compute intervals between packets:
    prev_time = np.zeros(mmap.num_types)
    intervals = [list() for _ in range(mmap.num_types)]
    for mark, t in zip(marks, timestamps):
        index_ = mark - 1
        intervals[index_].append(t - prev_time[index_])
        prev_time[index_] = t

    # Estimate mean values:
    mean_intervals = np.asarray([np.mean(ints) for ints in intervals])
    return np.power(mean_intervals, -1)


def estimate_ph_rate(ph: Ph, n_iters: int = 1000000):
    """
    Estimate PH distributions rates.
    """
    samples = [ph() for _ in range(n_iters)]
    return 1 / np.mean(samples)
