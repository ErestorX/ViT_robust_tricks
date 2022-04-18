import json

def test1():
    with open('output/val/all_summaries_test_with_head.json', 'r') as f:
        data = json.load(f)


def test2():
    with open('output/val/all_summaries_test.json', 'r') as f:
        data = json.load(f)

if __name__ == '__main__':
    test1()
    test2()