import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.evolutionary import hs
from opytimizer.spaces import search
from opytimizer.utils import constant

np.random.seed(0)


def test_hs_params():
    params = {"HMCR": 0.7, "PAR": 0.7, "bw": 10.0}

    new_hs = hs.HS(params=params)

    assert new_hs.HMCR == 0.7

    assert new_hs.PAR == 0.7

    assert new_hs.bw == 10.0


def test_hs_params_setter():
    new_hs = hs.HS()

    try:
        new_hs.HMCR = "a"
    except:
        new_hs.HMCR = 0.5

    try:
        new_hs.HMCR = -1
    except:
        new_hs.HMCR = 0.5

    assert new_hs.HMCR == 0.5

    try:
        new_hs.PAR = "b"
    except:
        new_hs.PAR = 0.5

    try:
        new_hs.PAR = -1
    except:
        new_hs.PAR = 0.5

    assert new_hs.PAR == 0.5

    try:
        new_hs.bw = "c"
    except:
        new_hs.bw = 5

    try:
        new_hs.bw = -1
    except:
        new_hs.bw = 5

    assert new_hs.bw == 5

    assert new_hs.bw == 5


def test_hs_generate_new_harmony():
    new_hs = hs.HS()

    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    agent = new_hs._generate_new_harmony(search_space.agents)

    assert agent.fit > 0


def test_hs_update():
    def square(x):
        return np.sum(x**2)

    new_hs = hs.HS()

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_hs.update(search_space, square)


def test_ihs_params():
    params = {"PAR_min": 0.5, "PAR_max": 1, "bw_min": 2, "bw_max": 5}

    new_ihs = hs.IHS(params=params)

    assert new_ihs.PAR_min == 0.5

    assert new_ihs.PAR_max == 1

    assert new_ihs.bw_min == 2

    assert new_ihs.bw_max == 5


def test_ihs_params_setter():
    new_ihs = hs.IHS()

    try:
        new_ihs.PAR_min = "a"
    except:
        new_ihs.PAR_min = 0.5

    try:
        new_ihs.PAR_min = -1
    except:
        new_ihs.PAR_min = 0.5

    assert new_ihs.PAR_min == 0.5

    try:
        new_ihs.PAR_max = "b"
    except:
        new_ihs.PAR_max = 1.0

    try:
        new_ihs.PAR_max = -1
    except:
        new_ihs.PAR_max = 1.0

    try:
        new_ihs.PAR_max = 0
    except:
        new_ihs.PAR_max = 1.0

    assert new_ihs.PAR_max == 1.0

    try:
        new_ihs.bw_min = "c"
    except:
        new_ihs.bw_min = 1.0

    try:
        new_ihs.bw_min = -1
    except:
        new_ihs.bw_min = 1.0

    assert new_ihs.bw_min == 1.0

    try:
        new_ihs.bw_max = "d"
    except:
        new_ihs.bw_max = 10.0

    try:
        new_ihs.bw_max = -1
    except:
        new_ihs.bw_max = 10.0

    try:
        new_ihs.bw_max = 0
    except:
        new_ihs.bw_max = 10.0

    assert new_ihs.bw_max == 10.0


def test_ihs_update():
    def square(x):
        return np.sum(x**2)

    new_ihs = hs.IHS()

    search_space = search.SearchSpace(
        n_agents=20, n_variables=2, lower_bound=[0, 0], upper_bound=[5, 5]
    )

    new_ihs.update(search_space, square, 1, 10)


def test_ghs_generate_new_harmony():
    new_ghs = hs.GHS()

    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    agent = new_ghs._generate_new_harmony(search_space.agents)

    assert agent.fit > 0

    new_ghs.HMCR = 0

    agent = new_ghs._generate_new_harmony(search_space.agents)

    assert agent.fit > 0


def test_sghs_params():
    params = {"LP": 100, "HMCRm": 0.98, "PARm": 0.9, "bw_min": 1, "bw_max": 10}

    new_sghs = hs.SGHS(params=params)

    assert new_sghs.LP == 100

    assert new_sghs.HMCRm == 0.98

    assert new_sghs.PARm == 0.9

    assert new_sghs.bw_min == 1

    assert new_sghs.bw_max == 10


def test_sghs_params_setter():
    new_sghs = hs.SGHS()

    try:
        new_sghs.HMCR = "a"
    except:
        new_sghs.HMCR = 0.5

    assert new_sghs.HMCR == 0.5

    try:
        new_sghs.PAR = "b"
    except:
        new_sghs.PAR = 0.5

    assert new_sghs.PAR == 0.5

    try:
        new_sghs.LP = 0.0
    except:
        new_sghs.LP = 100

    try:
        new_sghs.LP = 0
    except:
        new_sghs.LP = 100

    assert new_sghs.LP == 100

    try:
        new_sghs.HMCRm = "a"
    except:
        new_sghs.HMCRm = 0.98

    try:
        new_sghs.HMCRm = -1
    except:
        new_sghs.HMCRm = 0.98

    assert new_sghs.HMCRm == 0.98

    try:
        new_sghs.PARm = "b"
    except:
        new_sghs.PARm = 0.9

    try:
        new_sghs.PARm = -1
    except:
        new_sghs.PARm = 0.9

    assert new_sghs.PARm == 0.9

    try:
        new_sghs.bw_min = "c"
    except:
        new_sghs.bw_min = 1.0

    try:
        new_sghs.bw_min = -1
    except:
        new_sghs.bw_min = 1.0

    assert new_sghs.bw_min == 1.0

    try:
        new_sghs.bw_max = "d"
    except:
        new_sghs.bw_max = 10.0

    try:
        new_sghs.bw_max = -1
    except:
        new_sghs.bw_max = 10.0

    try:
        new_sghs.bw_max = 0
    except:
        new_sghs.bw_max = 10.0

    assert new_sghs.bw_max == 10.0


def test_sghs_compile():
    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_sghs = hs.SGHS()
    new_sghs.compile(search_space)

    try:
        new_sghs.lp = "a"
    except:
        new_sghs.lp = 1

    assert new_sghs.lp == 1

    try:
        new_sghs.lp = -1
    except:
        new_sghs.lp = 1

    assert new_sghs.lp == 1

    try:
        new_sghs.HMCR_history = "a"
    except:
        new_sghs.HMCR_history = []

    assert new_sghs.HMCR_history == []

    try:
        new_sghs.PAR_history = "a"
    except:
        new_sghs.PAR_history = []

    assert new_sghs.PAR_history == []


def test_sghs_generate_new_harmony():
    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_sghs = hs.SGHS()
    new_sghs.compile(search_space)

    agent = new_sghs._generate_new_harmony(search_space.agents)

    assert agent.fit > 0

    new_sghs.HMCR = 0

    agent = new_sghs._generate_new_harmony(search_space.agents)

    assert agent.fit > 0


def test_sghs_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_sghs = hs.SGHS()
    new_sghs.compile(search_space)

    new_sghs.update(search_space, square, 1, 10)
    new_sghs.update(search_space, square, 6, 10)

    new_sghs.lp = 1
    new_sghs.LP = 1

    new_sghs.update(search_space, square, 1, 10)


def test_nghs_params():
    params = {"pm": 0.1}

    new_nghs = hs.NGHS(params=params)

    assert new_nghs.pm == 0.1


def test_nghs_params_setter():
    new_nghs = hs.NGHS()

    try:
        new_nghs.pm = "a"
    except:
        new_nghs.pm = 0.1

    assert new_nghs.pm == 0.1

    try:
        new_nghs.pm = -1
    except:
        new_nghs.pm = 0.1

    assert new_nghs.pm == 0.1


def test_nghs_generate_new_harmony():
    new_nghs = hs.NGHS()
    new_nghs.pm = 1

    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    agent = new_nghs._generate_new_harmony(
        search_space.agents[0], search_space.agents[-1]
    )

    assert agent.fit > 0


def test_nghs_update():
    def square(x):
        return np.sum(x**2)

    new_nghs = hs.NGHS()

    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_nghs.update(search_space, square)

    assert search_space.agents[0].fit > 0


def test_goghs_generate_opposition_harmony():
    new_goghs = hs.GOGHS()

    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    agent = new_goghs._generate_opposition_harmony(
        search_space.agents[0], search_space.agents
    )

    assert agent.fit > 0


def test_goghs_update():
    def square(x):
        return np.sum(x**2)

    new_goghs = hs.GOGHS()

    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_goghs.update(search_space, square)

    assert search_space.agents[0].fit > 0
