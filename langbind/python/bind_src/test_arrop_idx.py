"""临时验证脚本: 测试 core.arrop.arr 的 [] 重载与 len() 支持

运行方式(构建产物 core.pyd 与本脚本同目录):
    python test_arrop_idx.py
"""
import core


def approx(a, b, eps=1e-3):
    return abs(a - b) < eps


def main():
    arr = core.arrop.arr(8)
    print("len(arr) =", len(arr))
    assert len(arr) == 8, "len() 应返回 8"

    for i in range(8):
        arr[i] = i * 0.5
    print("tolist  =", arr.tolist())

    got = [arr[i] for i in range(8)]
    print("getitem =", got)
    for i in range(8):
        assert approx(got[i], i * 0.5), f"idx {i} 读取错误: {got[i]}"

    assert approx(arr[-1], 3.5), "负索引读取错误"
    arr[-1] = 99.0
    assert approx(arr[7], 99.0), "负索引写入错误"
    print("negidx ok: arr[-1] =", arr[-1])

    sl = arr[1:4]
    print("slice[1:4] =", sl, type(sl).__name__)
    assert isinstance(sl, list)
    assert approx(sl[0], 0.5) and approx(sl[2], 1.5)

    sl2 = arr[::2]
    print("slice[::2] =", sl2)
    assert approx(sl2[0], 0.0) and approx(sl2[1], 1.0)

    arr[0] = 7
    assert approx(arr[0], 7.0), "int 写入错误"

    try:
        _ = arr[100]
        raise AssertionError("越界读取未抛异常")
    except IndexError:
        print("IndexError (read oob) ok")

    try:
        arr[100] = 1.0
        raise AssertionError("越界写入未抛异常")
    except IndexError:
        print("IndexError (write oob) ok")

    try:
        _ = arr["x"]
        raise AssertionError("非法 key 未抛异常")
    except TypeError:
        print("TypeError (bad key) ok")

    arr2 = core.arrop.arr(4)
    arr2.fromlist([1.0, 2.0, 3.0, 4.0])
    assert arr2.tolist() == [1.0, 2.0, 3.0, 4.0], arr2.tolist()
    print("fromlist/tolist ok:", arr2.tolist())

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
