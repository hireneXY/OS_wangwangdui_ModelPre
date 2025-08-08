import subprocess
import os
import streamlit as st

class KernelTuner:
    @staticmethod
    def check_root_permission():
        """检查是否有root权限"""
        return os.geteuid() == 0
    
    @staticmethod
    def get_kernel_param(param):
        """获取当前内核参数值"""
        try:
            result = subprocess.check_output(
                f"sysctl {param}", 
                shell=True, 
                text=True,
                stderr=subprocess.PIPE
            )
            return result.split("=")[1].strip()
        except subprocess.CalledProcessError as e:
            # 处理stderr，可能是bytes或字符串
            if e.stderr:
                if isinstance(e.stderr, bytes):
                    error_msg = e.stderr.decode('utf-8', errors='ignore')
                else:
                    error_msg = str(e.stderr)
            else:
                error_msg = str(e)
            st.warning(f"获取参数 {param} 失败: {error_msg}")
            return None
        except Exception as e:
            st.warning(f"获取参数 {param} 时发生未知错误: {str(e)}")
            return None

    @staticmethod
    def set_kernel_param(param, value):
        """设置内核参数（需root权限）"""
        # 检查root权限
        if not KernelTuner.check_root_permission():
            st.error(f"❌ 设置参数 {param} 失败: 需要root权限")
            st.info("💡 解决方案: 使用 'sudo streamlit run load_test_dashboard.py' 启动应用")
            return False
            
        try:
            # 对于包含空格的参数值，需要用引号包围
            if isinstance(value, str) and ' ' in value:
                cmd = f'sysctl -w {param}="{value}"'
            else:
                cmd = f"sysctl -w {param}={value}"
            
            result = subprocess.check_output(
                cmd, 
                shell=True, 
                text=True,
                stderr=subprocess.PIPE
            )
            st.success(f"✅ 成功设置参数 {param} = {value}")
            return True
        except subprocess.CalledProcessError as e:
            # 处理stderr，可能是bytes或字符串
            if e.stderr:
                if isinstance(e.stderr, bytes):
                    error_msg = e.stderr.decode('utf-8', errors='ignore')
                else:
                    error_msg = str(e.stderr)
            else:
                error_msg = str(e)
            st.error(f"❌ 设置参数 {param} = {value} 失败: {error_msg}")
            return False
        except Exception as e:
            st.error(f"❌ 设置参数 {param} 时发生未知错误: {str(e)}")
            return False

    @staticmethod
    def reset_default_params():
        """重置关键参数为默认值（测试结束后使用）"""
        if not KernelTuner.check_root_permission():
            st.error("❌ 重置参数失败: 需要root权限")
            st.info("💡 解决方案: 使用 'sudo streamlit run load_test_dashboard.py' 启动应用")
            return False
            
        default_params = {
            "vm.swappiness": 60,
            "net.ipv4.tcp_rmem": "4096 87380 6291456",
            "vm.dirty_ratio": 20,
            "vm.dirty_background_ratio": 10,
            "net.ipv4.tcp_wmem": "4096 16384 4194304"
        }
        
        success_count = 0
        total_count = len(default_params)
        
        for param, value in default_params.items():
            if KernelTuner.set_kernel_param(param, value):
                success_count += 1
        
        if success_count == total_count:
            st.success(f"✅ 所有参数重置成功 ({success_count}/{total_count})")
            return True
        else:
            st.warning(f"⚠️ 部分参数重置失败 ({success_count}/{total_count})")
            return False
    
    @staticmethod
    def get_current_user_info():
        """获取当前用户信息"""
        try:
            username = subprocess.check_output("whoami", shell=True, text=True).strip()
            uid = os.geteuid()
            return username, uid
        except:
            return "unknown", os.geteuid()
    