def skip_execution(skip=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not skip:
                return func(*args, **kwargs)
            # 如果skip为True，则不执行函数，返回None或提示
            print(f"函数 {func.__name__} 已跳过执行")
            return None
        return wrapper
    return decorator


def train_execution(train=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if train:
                return func(*args, **kwargs)
            # 如果skip为True，则不执行函数，返回None或提示
            print(f"跳过训练 {func.__name__} 已跳过执行")
            return None
        return wrapper
    return decorator
