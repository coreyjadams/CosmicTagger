import pytest
import random

def pytest_addoption(parser):
    parser.addoption('--limit', action='store', default=-1, type=int, help='tests limit')


def pytest_collection_modifyitems(session, config, items):
    limit = config.getoption('--limit')
    essential = []
    non_essential = []
    for item in items:
        if 'build_network' not in item.name:
            essential.append(item)
        else:
            non_essential.append(item)
    random.shuffle(non_essential)
    if limit < len(essential):
        items[:] = essential
    else:
        n = limit - len(essential)
        items[:] = essential + non_essential[:n]
    # if limit >= 0:
    #     print(type(items[0]))
    #     print(items[0])
    #     print(items[0].__dict__)
    #     items[:] = items[:limit]


# @pytest.fixture(scope="session", autouse=True)
# def callattr_ahead_of_alltests(request):
#     print("callattr_ahead_of_alltests called")
#     seen = {None}
#     session = request.node
#     print(seen)
#     for item in session.items:
#         cls = item.getparent(pytest.Class)
#         if cls not in seen:
#             if hasattr(cls.obj, "callme"):
#                 cls.obj.callme()
#             seen.add(cls)