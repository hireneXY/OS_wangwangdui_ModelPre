import os
import random
import time
import shutil
from tempfile import TemporaryDirectory

class MemoryWorkload:
    def run_memory_intensive(self, difficulty):
        """运行内存密集型负载"""
        # 根据难度设置参数
        if difficulty == "easy":
            size_mb = random.randint(100, 200)
            operations = 5000
        elif difficulty == "medium":
            size_mb = random.randint(300, 500)
            operations = 10000
        elif difficulty == "hard":
            size_mb = random.randint(600, 800)
            operations = 20000
        elif difficulty == "extreme":
            size_mb = random.randint(900, 1200)
            operations = 30000
        else:
            raise ValueError(f"未知难度级别: {difficulty}")
        
        # 分配内存并进行操作
        data = bytearray(size_mb * 1024 * 1024)  # 分配指定大小的内存
        start_time = time.time()
        
        # 随机访问内存
        for _ in range(operations):
            index = random.randint(0, len(data) - 1)
            data[index] = random.randint(0, 255)
        
        runtime = time.time() - start_time
        return runtime, size_mb

    def run_memory_intensive_custom(self, memory_size_mb, operations):
        """运行自定义内存密集型负载"""
        # 分配内存并进行操作
        data = bytearray(memory_size_mb * 1024 * 1024)  # 分配指定大小的内存
        start_time = time.time()
        
        # 随机访问内存
        for _ in range(operations):
            index = random.randint(0, len(data) - 1)
            data[index] = random.randint(0, 255)
        
        runtime = time.time() - start_time
        return runtime, memory_size_mb


class IOWorkload:
    def __init__(self):
        self.temp_dir = TemporaryDirectory()
        
    def run_io_intensive(self, difficulty):
        """运行IO密集型负载"""
        if difficulty == "easy":
            num_files = random.randint(3, 7)
            file_size_mb = random.randint(5, 20)
            read_operations = 20
        elif difficulty == "medium":
            num_files = random.randint(8, 12)
            file_size_mb = random.randint(30, 50)
            read_operations = 50
        elif difficulty == "hard":
            num_files = random.randint(13, 18)
            file_size_mb = random.randint(60, 80)
            read_operations = 100
        elif difficulty == "extreme":
            num_files = random.randint(19, 25)
            file_size_mb = random.randint(90, 120)
            read_operations = 150
        else:
            raise ValueError(f"未知难度级别: {difficulty}")
        
        start_time = time.time()
        file_paths = []
        
        # 写入文件
        for i in range(num_files):
            file_path = os.path.join(self.temp_dir.name, f"test_file_{i}.bin")
            file_paths.append(file_path)
            
            # 写入随机数据
            with open(file_path, 'wb') as f:
                # 分块写入大文件
                chunk_size = 1024 * 1024  # 1MB块
                remaining = file_size_mb * 1024 * 1024
                while remaining > 0:
                    write_size = min(chunk_size, remaining)
                    f.write(os.urandom(write_size))
                    remaining -= write_size
        
        # 随机读取文件
        total_read = 0
        for _ in range(read_operations):
            file_path = random.choice(file_paths)
            with open(file_path, 'rb') as f:
                # 随机位置读取
                f.seek(random.randint(0, os.path.getsize(file_path) - 1024))
                data = f.read(1024)
                total_read += len(data)
        
        runtime = time.time() - start_time
        total_read_mb = total_read / (1024 * 1024)
        total_write_mb = num_files * file_size_mb
        
        return runtime, num_files, total_read_mb, total_write_mb

    def run_io_intensive_custom(self, num_files, file_size_mb, read_operations):
        """运行自定义IO密集型负载"""
        start_time = time.time()
        file_paths = []
        
        # 写入文件
        for i in range(num_files):
            file_path = os.path.join(self.temp_dir.name, f"test_file_{i}.bin")
            file_paths.append(file_path)
            
            # 写入随机数据
            with open(file_path, 'wb') as f:
                # 分块写入大文件
                chunk_size = 1024 * 1024  # 1MB块
                remaining = file_size_mb * 1024 * 1024
                while remaining > 0:
                    write_size = min(chunk_size, remaining)
                    f.write(os.urandom(write_size))
                    remaining -= write_size
        
        # 随机读取文件
        total_read = 0
        for _ in range(read_operations):
            file_path = random.choice(file_paths)
            with open(file_path, 'rb') as f:
                # 随机位置读取
                f.seek(random.randint(0, os.path.getsize(file_path) - 1024))
                data = f.read(1024)
                total_read += len(data)
        
        runtime = time.time() - start_time
        total_read_mb = total_read / (1024 * 1024)
        total_write_mb = num_files * file_size_mb
        
        return runtime, num_files, total_read_mb, total_write_mb
    
    def run_file_cache_intensive(self, difficulty):
        """运行文件缓存密集型负载"""
        if difficulty == "easy":
            num_files = random.randint(300, 800)
            file_size_kb = random.randint(10, 50)
            access_operations = 500
        elif difficulty == "medium":
            num_files = random.randint(801, 1500)
            file_size_kb = random.randint(60, 100)
            access_operations = 1000
        elif difficulty == "hard":
            num_files = random.randint(1501, 2500)
            file_size_kb = random.randint(110, 200)
            access_operations = 1500
        elif difficulty == "extreme":
            num_files = random.randint(2501, 4000)
            file_size_kb = random.randint(210, 300)
            access_operations = 2000
        else:
            raise ValueError(f"未知难度级别: {difficulty}")
        
        # 创建子目录以避免单个目录中文件过多
        subdirs = [os.path.join(self.temp_dir.name, f"subdir_{i}") for i in range(10)]
        for subdir in subdirs:
            os.makedirs(subdir, exist_ok=True)
        
        start_time = time.time()
        file_paths = []
        
        # 创建小文件
        for i in range(num_files):
            subdir = subdirs[i % len(subdirs)]
            file_path = os.path.join(subdir, f"cache_file_{i}.bin")
            file_paths.append(file_path)
            
            with open(file_path, 'wb') as f:
                f.write(os.urandom(file_size_kb * 1024))  # 写入指定大小的随机数据
        
        # 随机访问文件以测试缓存
        for _ in range(access_operations):
            file_path = random.choice(file_paths)
            with open(file_path, 'rb') as f:
                f.read()  # 读取整个整个文件到缓存
        
        runtime = time.time() - start_time
        return runtime, num_files, file_size_kb

    def run_file_cache_intensive_custom(self, num_files, file_size_kb, access_operations):
        """运行自定义文件缓存密集型负载"""
        # 创建子目录以避免单个目录中文件过多
        subdirs = [os.path.join(self.temp_dir.name, f"subdir_{i}") for i in range(10)]
        for subdir in subdirs:
            os.makedirs(subdir, exist_ok=True)
        
        start_time = time.time()
        file_paths = []
        
        # 创建小文件
        for i in range(num_files):
            subdir = subdirs[i % len(subdirs)]
            file_path = os.path.join(subdir, f"cache_file_{i}.bin")
            file_paths.append(file_path)
            
            with open(file_path, 'wb') as f:
                f.write(os.urandom(file_size_kb * 1024))  # 写入指定大小的随机数据
        
        # 随机访问文件以测试缓存
        for _ in range(access_operations):
            file_path = random.choice(file_paths)
            with open(file_path, 'rb') as f:
                f.read()  # 读取整个文件到缓存
        
        runtime = time.time() - start_time
        return runtime, num_files, file_size_kb
    
    def drop_caches(self):
        """清理系统缓存（需要root权限）"""
        if os.geteuid() == 0:
            try:
                with open('/proc/sys/vm/drop_caches', 'w') as f:
                    f.write('3')  # 清除页缓存、目录项和inode
                return True
            except Exception:
                return False
        return False
    
    def __del__(self):
        """清理临时文件"""
        try:
            self.temp_dir.cleanup()
        except:
            pass


class MixedWorkload:
    def __init__(self):
        self.memory_workload = MemoryWorkload()
        self.io_workload = IOWorkload()
    
    def run_mixed_workload(self, difficulty):
        """运行混合负载（内存+IO）"""
        if difficulty == "easy":
            memory_size_mb = random.randint(50, 150)
            num_files = random.randint(2, 4)
            file_size_mb = random.randint(10, 30)
            iterations = 20
        elif difficulty == "medium":
            memory_size_mb = random.randint(151, 250)
            num_files = random.randint(5, 6)
            file_size_mb = random.randint(31, 60)
            iterations = 40
        elif difficulty == "hard":
            memory_size_mb = random.randint(251, 400)
            num_files = random.randint(7, 8)
            file_size_mb = random.randint(61, 100)
            iterations = 60
        elif difficulty == "extreme":
            memory_size_mb = random.randint(401, 600)
            num_files = random.randint(9, 12)
            file_size_mb = random.randint(101, 150)
            iterations = 80
        else:
            raise ValueError(f"未知难度级别: {difficulty}")
        
        start_time = time.time()
        data = bytearray(memory_size_mb * 1024 * 1024)  # 分配内存
        
        with TemporaryDirectory() as temp_dir:
            # 创建初始文件
            file_paths = [os.path.join(temp_dir, f"mixed_file_{i}.bin") for i in range(num_files)]
            for file_path in file_paths:
                with open(file_path, 'wb') as f:
                    f.write(os.urandom(file_size_mb * 1024 * 1024))
            
            # 交替进行内存和IO操作
            for _ in range(iterations):
                # 内存操作
                for _ in range(100):
                    index = random.randint(0, len(data) - 1)
                    data[index] = random.randint(0, 255)
                
                # IO操作
                file_path = random.choice(file_paths)
                with open(file_path, 'r+b') as f:
                    f.seek(random.randint(0, os.path.getsize(file_path) - 1024))
                    f.write(os.urandom(1024))
        
        runtime = time.time() - start_time
        return runtime, memory_size_mb, num_files, file_size_mb
    