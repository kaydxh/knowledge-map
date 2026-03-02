# Linux

## 性能

### 性能工具图谱

-  

	- linux性能图谱.png

### 性能优化图谱

-  

	- 性能优化.png

## CPU性能

### 进程状态

- 可运行状态

	- 正在使用 CPU 或者正在等待 CPU 的进程，用 ps 命令看处于 R 状态（Running 或 Runnable）的进程

- 不可中断状态

	- 正处于内核态关键流程中的进程，并且这些流程是不可打断的，比如最常见的是等待硬件设备的 I/O 响应，也就是ps 命令中看到的 D 状态（Uninterruptible Sleep，也称为 Disk Sleep）的进程

	- 比如，当一个进程向磁盘读写数据时，为了保证数据的一致性，在得到磁盘回复前，它是不能被其他进程或者中断打断的，这个时候的进程就处于不可中断状态。如果此时的进程被打断了，就容易出现磁盘数据与进程数据不一致的问题

	- 不可中断状态实际上是系统对进程和硬件设备的一种保护机制

### 平均负载

- 定义

	- 平均负载是指单位时间内，系统处于可运行状态和不可中断状态的平均进程数，也就是平均活跃进程数

	- 包括了正在使用 CPU 的进程，还包括等待 CPU 和等待 I/O 的进程

- 工具

	- 查看cpu核数

		- grep 'model name' /proc/cpuinfo | wc -l

	- 查看负载

		- 
$ uptime
02:34:03 up 2 days, 20:14,  1 user,  load average: 0.63, 0.83, 0.88（1 分钟、5 分钟、15 分钟的平均负载）

		- 
# -d 参数表示高亮显示变化的区域
$ watch -d uptime

		- htop

	- 模拟平均负载升高

		- stress

			- 模拟一个 CPU 使用率 100% 的场景

				- # --cpu cpu压测选项，-i io压测选项，-c 进程数压测选项，--timeout 执行时间
$ stress --cpu 1 --timeout 600

			- 模拟 I/O 压力，即不停地执行 sync

				- $ stress -i 1 --timeout 600

				- iowait可能无法升高

					- 因为stress使用的是 sync() 系统调用，它的作用是刷新缓冲区内存到磁盘中。对于新安装的虚拟机，缓冲区可能比较小，无法产生大的IO压力，这样大部分就都是系统调用的消耗了。所以，会看到只有系统CPU使用率升高。

					- 解决方法是使用stress的下一代stress-ng，它支持更丰富的选项，比如 stress-ng -i 1 --hdd 1 --timeout 600（--hdd表示读写临时文件）

			- 模拟的是 8 个进程

				- $ stress -c 8 --timeout 600

	- 监控和分析系统的性能

		- 多核 CPU 性能分析工具

			- mpstat

				- # -P ALL 表示监控所有CPU，后面数字5表示间隔5秒后输出一组数据$ mpstat -P ALL 5

		- 进程性能分析工具

			- pidstat

				- # 间隔5秒后输出一组数据
$ pidstat -u 5 1
13:37:07      UID       PID    %usr %system  %guest   %wait    %CPU   CPU  Command
13:37:12        0      2962  100.00    0.00    0.00    0.00  100.00     1  stress

				- 注意：CentOS默认的sysstat稍微有点老，源码或者RPM升级到11.5.5版本以后就可以看到了iowait列

- 指标

	- 当平均负载高于 CPU 数量 70% 的时候，应该分析排查负载高的问题了。一旦负载过高，就可能导致进程响应变慢，进而影响服务的正常功能。

- 负载高的可能原因

	- cpu密集型导致负载高，状况是cpu使用率和负载同时变高

	- io密集型：iowait很高同时负载很高

	- 大量等待 CPU 的进程调度也会导致平均负载升高

- 排查步骤

	- 通过uptime／top看系统负载

	- 通过top命令中的%CPU 或者mpstat中的%idle %iowait %wait区分是cpu密集型还是io密集型任务还是大量进程等待调度导致

	- 通过pidstat辅助分析具体是哪个进程导致的

### CPU 上下文

- 定义

	- CPU 寄存器

		- CPU 内置的容量小、但速度极快的内存

	- 程序计数器

		- 用来存储 CPU 正在执行的指令位置、或者即将执行的下一条指令位置

- 上下文切换

	- 定义

		- 把前一个任务的 CPU 上下文（也就是 CPU 寄存器和程序计数器）保存起来，然后加载新任务的上下文到这些寄存器和程序计数器，最后再跳转到程序计数器所指的新位置，运行新任务

	- 耗时

		- 每次上下文切换都需要几十纳秒到数微秒的 CPU 时间

	- 场景

		- 进程上下文切换

			- 进程的运行空间

				- 定义

					- 内核空间

						- Ring 0：具有最高权限，可以直接访问所有资源

					- 用户空间

						- Ring 3： 只能访问受限资源，不能直接访问内存等硬件设备，必须通过系统调用陷入到内核中，才能访问这些特权资源

				- 用户态

					- 进程在用户空间运行时

				- 内核态

					- 进程在内核空间运行时

				- 从用户态到内核态的转变，需要通过系统调用来完成

					- 系统调用（特权模式切换）

						- 一次系统调用的过程，发生了两次 CPU 上下文切换

							- CPU 寄存器里原来用户态的指令位置，需要先保存起来。接着，为了执行内核态代码，CPU 寄存器需要更新为内核态指令的新位置。最后才是跳转到内核态运行内核任务

							- 系统调用结束后，CPU 寄存器需要恢复原来保存的用户态，然后再切换到用户空间，继续运行进程

							- 不会涉及到虚拟内存等进程用户态的资源，也不会切换进程

						- 进程上下文切换跟系统调用有什么区别

							- 进程是由内核来管理和调度的，进程的切换只能发生在内核态。所以，进程的上下文不仅包括了虚拟内存、栈、全局变量等用户空间的资源，还包括了内核堆栈、寄存器等内核空间的状态

							- 因此，进程的上下文切换就比系统调用时多了一步：在保存当前进程的内核状态和 CPU 寄存器之前，需要先把该进程的虚拟内存、栈等保存下来；而加载了下一进程的内核态后，还需要刷新进程的虚拟内存和用户栈

		- 线程上下文切换

			- 前后两个线程属于不同进程

				- 因为资源不共享，所以切换过程就跟进程上下文切换是一样

			- 前后两个线程属于同一个进程

				- 因为虚拟内存是共享的，所以在切换时，虚拟内存这些资源就保持不动，只需要切换线程的私有数据、寄存器等不共享的数据

		- 中断上下文切换

			- 为了快速响应硬件的事件，中断处理会打断进程的正常调度和执行，转而调用中断处理程序，响应设备事件

			- 跟进程上下文不同，中断上下文切换并不涉及到进程的用户态。所以，即便中断过程打断了一个正处在用户态的进程，也不需要保存和恢复这个进程的虚拟内存、全局变量等用户态资源。中断上下文，其实只包括内核态中断服务程序执行所必需的状态，包括 CPU 寄存器、内核堆栈、硬件中断参数等

			- 对同一个 CPU 来说，中断处理比进程拥有更高的优先级，所以中断上下文切换并不会与进程上下文切换同时发生

	- 工具

		- 系统总体的上下文切换情况

			- vmstat

				- 主要用来分析系统的内存使用情况，也常用来分析 CPU 上下文切换和中断的次数

					- 
# 每隔5秒输出1组数据
$ vmstat 5
procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----
 r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st
 0  0      0 7005360  91564 818900    0    0     0     0   25   33  0  0 100  0  0

						- cs（context switch）是每秒上下文切换的次数

						- in（interrupt）则是每秒中断的次数

						- r（Running or Runnable）是就绪队列的长度，也就是正在运行和等待 CPU 的进程数

							- 如果大于cpu个数，会产生 CPU 竞争

						- b（Blocked）则是处于不可中断睡眠状态的进程数

		- 每个进程的上下文切换情况

			- pidstat

				- 
# 每隔1秒输出1组数据（需要 Ctrl+C 才结束）
# -w参数表示输出进程切换指标，而-u参数则表示输出CPU使用指标
$ pidstat -w -u 1
08:06:33      UID       PID    %usr %system  %guest   %wait    %CPU   CPU  Command
08:06:34        0     10488   30.00  100.00    0.00    0.00  100.00     0  sysbench
08:06:34        0     26326    0.00    1.00    0.00    0.00    1.00     0  kworker/u4:2

08:06:33      UID       PID   cswch/s nvcswch/s  Command
08:06:34        0      4089      1.00      0.00  kworker/1:5
08:06:34        0      4333      1.00      0.00  kworker/0:3
08:06:34        0     10499      1.00    224.00  pidstat
08:06:34        0     26326    236.00      0.00  kworker/u4:2
08:06:34     1000     26784    223.00      0.00  sshd

					- cswch

						- 每秒自愿上下文切换（voluntary context switches）的次数

							- 所谓自愿上下文切换，是指进程无法获取所需资源，导致的上下文切换。比如说， I/O、内存等系统资源不足时，就会发生自愿上下文切换

					- nvcswch

						- 每秒非自愿上下文切换（non voluntary context switches）的次数

							- 非自愿上下文切换，则是指进程由于时间片已到等原因，被系统强制调度，进而发生的上下文切换。比如说，大量进程都在争抢 CPU 时，就容易发生非自愿上下文切换

					- CPU

						- cpu使用率

				- pidstat默认显示的是进程指标数据

					- 加上 -t 参数后，才会输出线程的指标

						- 
# 每隔1秒输出一组数据（需要 Ctrl+C 才结束）
# -wt 参数表示输出线程的上下文切换指标
$ pidstat -wt 1
08:14:05      UID      TGID       TID   cswch/s nvcswch/s  Command
...
08:14:05        0     10551         -      6.00      0.00  sysbench
08:14:05        0         -     10551      6.00      0.00  |__sysbench
08:14:05        0         -     10552  18911.00 103740.00  |__sysbench
08:14:05        0         -     10553  18915.00 100955.00  |__sysbench
08:14:05        0         -     10554  18827.00 103954.00  |__sysbench
...

							- sysbench的进程（主线程）的上下文切换次数看起来并不多，但它的子线程的上下文切换次数却有很多

		- 中断的变化情况

			- 
# -d 参数表示高亮显示变化的区域
$ watch -d cat /proc/interrupts
           CPU0       CPU1
...
RES:    2450431    5279697   Rescheduling interrupts
...

				- 重调度中断（RES）

					- 表示，唤醒空闲状态的 CPU 来调度新的任务运行

					- 如果RES中断指标升高是因为过多任务的调度导致的

		- 模拟系统多线程调度切换

			- sysbench

				- 模拟系统多线程调度的瓶颈

					- 
# 以10个线程运行5分钟的基准测试，模拟多线程切换的问题
$ sysbench --threads=10 --max-time=300 threads run

	- 指标

		- 总体指标

			- 如果系统的上下文切换次数比较稳定，那么从数百到一万以内，都应该算是正常的。但当上下文切换次数超过一万次，或者切换次数出现数量级的增长时，就很可能已经出现了性能问题

		- 具体指标

			- 自愿上下文切换变多

				- 说明进程都在等待资源，有可能发生了 I/O 等其他问题

			- 非自愿上下文切换变多

				- 说明进程都在被强制调度，也就是都在争抢 CPU，说明 CPU 的确成了瓶颈

			- 中断次数变多

				- 说明 CPU 被中断处理程序占用，还需要通过查看 /proc/interrupts 文件来分析具体的中断类型

	- 排查问题

		- 通过vmstat 1 产看cpu的上下文切换和中断情况

		- 通过pidstat -wt 1查看是哪个进程导致的

		- 通过watch -d cat /proc/interrupts观察中断的变化

### cpu使用率

- 定义

	- 就是除了空闲时间外的其他时间占总 CPU 时间的百分比

		-  

		- 性能工具一般都会取间隔一段时间（比如 3 秒）的两次值，作差后，再计算出这段时间内的平均 CPU 使用率

			-  

- cpu时间片

	- Linux 通过事先定义的节拍率（内核中表示为 HZ），触发时间中断，并使用全局变量 Jiffies 记录了开机以来的节拍数。每发生一次时间中断，Jiffies 的值就加 1

	- 节拍率 HZ 是内核的可配选项，可以设置为 100、250、1000 等。不同的系统可能设置不同数值

		- 查看系统节拍率

			- 内核

				- $ grep 'CONFIG_HZ=' /boot/config-$(uname -r)
CONFIG_HZ=250

			- 用户

				- 用户空间节拍率 USER_HZ，它总是固定为 100，也就是 1/100 秒

- 相关字段含义

	- user（通常缩写为 us）

		- 代表用户态 CPU 时间。注意，它不包括下面的 nice 时间，但包括了 guest 时间

	- nice（通常缩写为 ni）

		- 代表低优先级用户态 CPU 时间，也就是进程的 nice 值被调整为 1-19 之间时的 CPU 时间。这里注意，nice 可取值范围是 -20 到 19，数值越大，优先级反而越低

	- system（通常缩写为 sys）

		- 代表内核态 CPU 时间

	- idle（通常缩写为 id）

		- 代表空闲时间。注意，它不包括等待 I/O 的时间（iowait）。iowait（通常缩写为 wa），代表等待 I/O 的 CPU 时间

	- irq（通常缩写为 hi）

		- 代表处理硬中断的 CPU 时间

	- softirq（通常缩写为 si）

		- 代表处理软中断的 CPU 时间

	- steal（通常缩写为 st）

		- 代表当系统运行在虚拟机中的时候，被其他虚拟机占用的 CPU 时间

	- guest（通常缩写为 guest）

		- 代表通过虚拟化运行其他操作系统的时间，也就是运行虚拟机的 CPU 时间

	- guest_nice（通常缩写为 gnice）

		- 代表以低优先级运行虚拟机的时间

	- %wait

		- 等待 CPU 使用率

- 注意点

	- top 和 ps 这两个工具报告的 CPU 使用率不一样的原因

		- 因为 top 默认使用 3 秒时间间隔，而 ps 使用的却是进程的整个生命周期

- 命令

	- 获取系统的 CPU 和任务统计信息

		- 
# 只保留各个CPU的数据
$ cat /proc/stat | grep ^cpu
cpu  280580 7407 286084 172900810 83602 0 583 0 0 0
cpu0 144745 4181 176701 86423902 52076 0 301 0 0 0
cpu1 135834 3226 109383 86476907 31525 0 282 0 0 0

	- 获取每个进程提供了运行情况的统计信息

		- /proc/[pid]/stat，统计了进程的指标

	- 查看 CPU 使用率

		- top

			- 显示了系统总体的 CPU 和内存使用情况，以及各个进程的资源使用情况

				- Running(就绪队列running进程数）

				- 进程状态

					- R（running）

					- S(Sleep)

			- 没有细分用户态和内核态的使用率

		- ps

			- 显示了每个进程的资源使用情况

		- pidstat

			- 每隔1秒输出一组数据，共输出5组 
pidstat 1 5

	- 排查CPU 使用率过高问题

		- perf

			- perf top

				- 它能够实时显示占用 CPU 时钟最多的函数或者指令，可以用来查找热点函数

				- 
$ perf top
Samples: 833  of event 'cpu-clock', Event count (approx.): 97742399
Overhead  Shared Object       Symbol
   7.28%  perf                [.] 0x00000000001f78a4
   4.72%  [kernel]            [k] vsnprintf
   4.32%  [kernel]            [k] module_get_kallsym
   3.65%  [kernel]            [k] _raw_spin_unlock_irqrestore
...

					- 含义

						- 第一列 Overhead

							- 是该符号的性能事件在所有采样中的比例，用百分比来表示

						- 第二列 Shared

							- 是该函数或指令所在的动态共享对象（Dynamic Shared Object），如内核、进程名、动态链接库名、内核模块名等

						- 第三列 Object

							- 是动态共享对象的类型。比如 [.] 表示用户空间的可执行程序、或者动态链接库，而 [k] 则表示内核空间

						- 最后一列 Symbol

							- 是符号名，也就是函数名。当函数名未知时，用十六进制的地址来表示

				- 缺点

					- 不能保存数据

			- perf record

				- -g开启调用关系分析，-p指定进程号
perf top -g -p 进程号-o file.txt

			- perf report

				- 展示类似于perf top的报告
perf report -i file.txt

- 排查问题

	- 通过top、ps和pidstat找到cpu占用率很高的进程

	- 使用 perf 分析 CPU 性能问题

		- perf top

## 内存

### 内存泄漏

- 排查方法

	- 工具安装

		- [gperftools](https://github.com/gperftools/gperftools)

			- yum install libunwind

			- yum install pprof

			- yum install ghostscript 

		- 生成pdf依赖

			- yum -y install graphviz

			- yum -y install ghostscript

	- 环境变量

		- export LD_LIBRARY_PATH=$install_path/lib:$LD_LIBRARY_PATH
export LD_PRELOAD="libtcmalloc.so"
export HEAPPROFILESIGNAL=`kill -l SIGUSR1`
export HEAPPROFILE=test.pprof
export HEAP_PROFILE_ALLOCATION_INTERVAL=0
export HEAP_PROFILE_DEALLOCATION_INTERVAL=0
export HEAP_PROFILE_INUSE_INTERVAL=0
export HEAP_PROFILE_TIME_INTERVAL=0

	- 启动

		- # nohup <binary> 1>srv.nohup.out.$(date +%s) 2>&1&
nohup ./bin/stream_service --conf=./conf/stream_service.conf 1>srv.nohup.out.$(date +%s) 2>&1&

	- 触发信号进行内存profile文件导出

		- kill -s SIGUSR1 `pidof <binary>`

	- prof文件导出pdf

		- pprof --pdf <binary> test.pprof.0001.heap > export.pdf

	- 比较前后2次的内存差值

		- pprof --pdf  --base=test.pprof_93974.0001.heap ./bin/stream_service test.pprof_93974.0002.heap > diff.pdf

	- 针对go程序

		- # 获取基准heap 
curl -s http://127.0.0.1:30010/debug/pprof/heap > base.heap

		- # 压测后的heap
curl -s http://127.0.0.1:30010/debug/pprof/heap > current.heap

		- # 比较2次的内存差值
go tool pprof --pdf  --base base.heap current.heap > diff.pdf

	- 工具检测

		- AddressSanitizer

			- --extra-cflags=' -O0 -g3 -fsanitize=address -Wno-error -fPIC -I/usr/local/include' --extra-ldflags='-O0 -g3 -fsanitize=address -Wno-error -fPIC '


		- Valgrind

			- /usr/bin/valgrind -v --tool=memcheck --gen-suppressions=all  --leak-check=full --show-leak-kinds=all --leak-resolution=med --track-origins=yes --log-file=mem_leak.log test_exec

## I/O

### linux文件系统

- 架构图

	-  

		- fs.png

- 为每个文件分配2个数据结构

	- 索引节点和目录项

		- 索引节点（index node）

			- 记录文件的元数据，比如 inode 编号、文件大小、访问权限、修改日期、数据的位置等

			- 索引节点和文件一一对应，它跟文件内容一样，都会被持久化存储到磁盘中。索引节点同样占用磁盘空间

		- 目录项（dentry）

			- 用来记录文件的名字、索引节点指针以及与其他目录项的关联关系，目录项本身是一个内存数据结构

		- 索引节点和目录项区别

			- 索引节点是每个文件的唯一标志，而目录项维护的正是文件系统的树状结构

			- 目录项和索引节点的关系是多对一，即一个文件可以有多个别名

		- 磁盘

			- 文件系统格式化时，分成三个存储区域

				- 超级块

					- 存储整个文件系统的状态

				- 索引节点区

					- 用来存储索引节点

				- 数据块区

					- 用来存储文件数据

			- 读写的最小单位是扇区

			- 文件系统的最小读写单元是逻辑块

				- 常见的逻辑块大小为 4KB，也就是由连续的 8 个扇区组成

		- 架构图

			-  

				- 索引磁盘.png

- 虚拟文件系统VFS（Virtual File System）

	- 作用

		- VFS 定义了一组所有文件系统都支持的数据结构和标准接口

		- 屏蔽底层文件系统

			- 磁盘文件系统

				- Ext4

				- XFS

				- OverlayFS

			- 内存文件系统

				- /proc 文件系统

				- /sys 文件系统

			- 网络文件系统

				- 类别

					- NFS

					- SMB

					- iSCSI

				- 访问其他计算机数据的文件系统

- 文件系统IO

	- 分类

		- 是否利用标准库缓存

			- 缓冲I/O

				- 利用标准库缓存来加速文件的访问，标准库内部再通过系统调度访问文件

			- 非缓冲I/O

				- 直接通过系统调用来访问文件，不再经过标准库缓存

		- 是否利用操作系统的页缓存

			- 直接 I/O

				- 是指跳过操作系统的页缓存，直接跟文件系统交互来访问文件（系统调用指定 O_DIRECT 标志）

			- 非直接 I/O

				- 经过系统的页缓存，然后再由内核或额外的系统调用，真正写入磁盘（默认非直接 I/O）

		- 根据应用程序是否阻塞自身运行

			- 阻塞 I/O

				- 应用程序执行 I/O 操作后，如果没有获得响应，就会阻塞当前线程，自然就不能执行其他任务

			- 非阻塞 I/O

				- 应用程序执行 I/O 操作后，不会阻塞当前的线程，可以继续执行其他的任务，随后再通过轮询或者事件通知的形式，获取调用的结果（设置 O_NONBLOCK 标志）

		- 根据是否等待响应结果

			- 同步I/O

				- 应用程序执行 I/O 操作后，要一直等到整个 I/O 完成后，才能获得 I/O 响应(设置了 O_SYNC 或者 O_DSYNC 标志)

			- 异步 I/O

				- 应用程序执行 I/O 操作后，不用等待完成和完成后的响应，而是继续执行就可以。等到这次 I/O 完成后，响应会用事件通知的方式，告诉应用程序(设置了 O_ASYNC 选项，内核会再通过 SIGIO 或者 SIGPOLL，来通知进程文件是否可读写)

## 网络部分

### DNS(Domain name system)

- DNS特点

	- 以分层结构管理

		- 以点分开，位置越后层级越高，如：acx.gk.org
一级域名：org
二级域名：gk
三级域名：acx

		- 域名解析也是从1级域名开始，发送给每个层级的域名服务器，直到解析完成

	- 提供域名和IP地址映射的关系查询服务

	- 动态服务发现和全局负载均衡（Global Server Load Balance，GSLB）

- 协议

	- DNS 协议在 TCP/IP 栈中属于应用层

	- 传输层UDP居多，也有TCP

- DNS查询

	- 查询原理

		- 缓存命中，直接返回

		- 否则，从一级开始递归查询

	- 配置文件

		- 外网域名解析

			- /etc/resolv.conf 

		- 内网域名解析

			- /etc/hosts

	- 工具

		- nslookup

			- nslookup acx.gk.org

		- dig（提供了trace功能，可以展示递归查询的整个过程）

			- 1. +trace表示开启跟踪查询 
2. +nodnssec表示禁止DNS安全扩展
dig +trace +nodnssec acx.gk.org

	- 问题

		- DNS解析失败

			- 
/# nslookup -debug time.geekbang.org
;; Connection to 127.0.0.1#53(127.0.0.1) for time.geekbang.org failed: connection refused.
;; Connection to ::1#53(::1) for time.geekbang.org failed: address not available.

				- 可能原因

					- 检查/etc/resolv.conf里的dns服务器是否正确

						- 如果没有增加域名服务器地址：
echo "nameserver 114.114.114.114" > /etc/resolv.conf

		- DNS 解析不稳定，解析时间忽快忽慢

			- 可能原因

				- DNS 服务器本身有问题，响应慢并且不稳定

				- 客户端到 DNS 服务器的网络延迟比较大

					- ping下DNS服务器（如ping -c3 8.8.8.8），检查下延迟时间是否大（一般大几十ms就比较大了）

						- 如果延迟大，可以考虑更改DNS服务器地址，如114.114.114.114

						- 如果没有使用DNS缓存

							- # 使用dnsmasq，开启dsn缓存
 /etc/init.d/dnsmasq start
 * Starting DNS forwarder and DHCP server dnsmasq                    [ OK ]

				- DNS 请求或者响应包，在某些情况下被链路中的网络设备弄丢了

		- 服务重启后，resolv.conf文件内容恢复默认

			- 1. cat /etc/sysconfig/network-scripts/ifcfg-xxx
2. NM_CONTROLLED="no"  //是否允许Network Manager管理，设置为no

## [内核](https://xinqiu.gitbooks.io/linux-insides-cn/content/Misc/linux-misc-4.html)

### 中断

- 定义

	- 是指处理器接收到来自硬件或软件的信号，提示发生了某个事件，应该被注意，这种情况就称为中断

- 组成

	- 上半部

		- 直接处理硬件请求，即硬中断，特点是快速执行

	- 下半部

		- 定义

			- 由内核触发，即软中断，特点是延迟执行

		- 实现方式

			- 软中断，可在所有处理器上同时执行，同类型也可以，仅网络和SCSI直接使用

			- tasklet，通过软中断实现，同类型不能在处理器上同时执行，大部分下半部处理又tasklet实现

			- 工作队列，在进程上下文中执行，允许重新调度甚至睡眠，如获得大量内存、信号量、执行阻塞式I/O非常有用

			- 子主题 4

- 特点

	- 中断是一种异步的事件处理机制，能提高系统的并发处理能力

	- 为了减少对正常进程运行进行影响，中断处理程序需要尽快运行

### 用户态到内核态切换

### 设备驱动

### read系统调用实现

### 内核参数

- # 文件包含限制一个进程可以拥有的VMA(虚拟内存区域)的数量
# 当这个值太小，不够用时，操作系统可能抛出内存不足的错误
max_map_count

	- 修改值

		- # 方法一
sysctl -w vm.max_map_count=65535

		- # 方法二
echo 65535 > /proc/sys/vm/max_map_count

### [加载运行可执行程序的过程](https://xuezhaojiang.github.io/LinuxKernel/lab7/lab7.html)

- 程序加载

	- 将可执行目标文件中的代码和数据从磁盘拷贝到存储器中，然后通过跳转到程序的第一条指令或入口点，也就是也就是符号_start的地址

## 命令

### linux常用命令

- 文件操作

	- 查找/统计

		- grep

			- 显示文件名

				- grep --with-filename  "key"   server.20211202160000.log* 

		- 输出json文件中某个key的value

			- # 如果key为多层，就采用多维数组，如['reflect']['image']
cat test.json | python -c "import sys, json; print(json.dumps(json.load(sys.stdin)['action_video'], indent=2))"

		- awk

			-  cat *.meta | grep -i 'terminal_trace_id' -A 1  | grep  [0-9] | awk '{for(i=1;i<=NF;++i) printf $i "\n";}' 

			- 文件里每行开头增加行号

				- # $0表示原来每行的内容
# NR表示行号
# 双引号之间表示行号与原来内容之间的分隔符
awk '$0=NR":"$0' filename

		- sed

			- 替换

				- # 将\"替换为”
echo '"{\"image_decode_ret\":0"}' | sed 's#\\"#"#g'

				- # 将\\n替换为真正的空行
echo '"{\"image_decode_ret\":0"}\n' | sed 's#\\"#"#g' | sed 's#\\n#\n#g'

	- 空间占用

		- du

			- 显示目录占用的磁盘空间大小, 目录深度为1

				- du -ah --max-depth=1

			- 参数

				- -h

					- 以可读方式显示

				- -a

					- 显示目录占用的磁盘空间大小，还要显示其下目录和文件占用磁盘空间的大小

				- -s

					- 显示目录占用的磁盘空间大小，不要显示其下子目录和文件占用的磁盘空间大小

				- -c

					- 显示几个目录或文件占用的磁盘空间大小，还要统计它们的总和--apparent-size：显示目录或文件自身的大小

				- -l

					- 统计硬链接占用磁盘空间的大小(在统计目录占用磁盘空间大小时，-l选项会把硬链接也统计进来)

				- -L

					- 统计符号链接所指向的文件占用的磁盘空间大小

	- 文件夹递归md5

		- find . -type f -exec md5sum {} \; 

- 磁盘

	- 空间占用

		- df

			- 统计当前文件所在的文件系统信息

				- df -hT .

	- inode使用

		- df -i

### cpu

- 利用率

	- 步骤

		- top

		- pidstat -p xxx

		- pstree

		- # 记录性能事件，查看报告
perf record -ag -- sleep 2;perf report

		- # 按 Ctrl+C 结束
execsnoop

	- 可能到原因

		- 大量的xx进程在启动时初始化失败，进而导致用户 CPU 使用率的升高

- cpu型号

	- cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c

- cpu核数/线程

	- lscpu | grep -E '^Thread|^Core|^Socket|^CPU\('

- cpu详细信息

	-  cat /proc/cpuinfo 

### gpu

- 利用率

	- 精确到单个进程

		- watch -n 1 nvidia-smi pmon -c 1 

	- 总的利用率

		- watch -n 1 nvidia-smi

### 内存

- 内存带宽

	- 官方源码地址：http://www.cs.virginia.edu/stream/FTP/Code/stream.c

	- 编译：
gcc -O3 -fopenmp -DN=2000000 -DNTIMES=10 stream.c -o stream

	- 指定运行线程为4
export OMP_NUM_THREADS=4

	- ./stream

	-  

- 查看内存条个数和大小

	- dmidecode |grep -P -A 5 "Memory Device"|grep Size|grep -v 'Range'

- 查看内存条详细信息（内存容量、内存频率、频率）

	- dmidecode

		-  

	- [带宽计算公式](https://zhuanlan.zhihu.com/p/443104177)

		- 带宽 ＝ 频率 * 位宽／8
2933 * 64 /8 = 23.464GB/s

### I/O

- 性能指标与检测

	- 文件系统io性能指标

		- 存储空间容量、使用量以及剩余空间

			- df

				- df -h /dev/sda1

		- 索引节点容量、使用量以及剩余量

			- df

				- df -i /dev/sda1

		- 缓存指标

			- 页缓存

				- cat /proc/meminfo | grep -E "SReclaimable|Cached"

			- 目录项缓存

			- 索引节点缓存

			- 具体文件系统缓存（如ext4的缓存）

				- # 按下c按照缓存大小排序，按下a按照活跃对象数排序
slabtop

		- IOPS（文件IO）

		- 响应时间

		- 吞吐量（B/s)

	- 问题

		- 如果磁盘空间足够，但是仍有空间不足的问题，可能是索引节点空间不足

			- 解决方案

				- 删除这些小文件，或者把它们移动到索引节点充足的其他磁盘中

	- 磁盘io性能指标与检测

		- 磁盘io性能指标

			- 使用率

				- 磁盘忙处理 I/O 请求的百分比。过高的使用率（比如超过 60%）通常意味着磁盘 I/O 存在性能瓶颈

			- IOPS（Input/Output Per Second）

				- 每秒的 I/O 请求数

			- 吞吐量

				- 每秒的 I/O 请求大小

			- 响应时间

				- 从发出 I/O 请求到收到响应的间隔时间

			- 缓冲器（Buffer)

		- 检测方法

			- 查看系统的 CPU 使用情况，关注 iowait

				- top

			- 检测磁盘的 I/O 使用率

				- iostat

			- 查看大量 I/O 的进程

				- pidstat

				- iotop

			- 找出问题进程正在读写的文件

				- 系统调用工具

					- strace

					- filetop

					- opensnoop

				- lsof

			- 思路图

				-  

					- io.PNG

	- 磁盘/文件系统io基准测试

		- fio

			- 测试项

				- 随机读

					- fio -name=randread -direct=1 -iodepth=64 -rw=randread -ioengine=libaio -bs=4k -size=1G -numjobs=1 -runtime=1000 -group_reporting -filename=/dev/sdb

				- 随机写

					- fio -name=randwrite -direct=1 -iodepth=64 -rw=randwrite -ioengine=libaio -bs=4k -size=1G -numjobs=1 -runtime=1000 -group_reporting -filename=/dev/sdb

				- 顺序读

					- fio -name=read -direct=1 -iodepth=64 -rw=read -ioengine=libaio -bs=4k -size=1G -numjobs=1 -runtime=1000 -group_reporting -filename=/dev/sdb

				- 顺序写

					- fio -name=write -direct=1 -iodepth=64 -rw=write -ioengine=libaio -bs=4k -size=1G -numjobs=1 -runtime=1000 -group_reporting -filename=/dev/sdb

			- 重点参数

				- direct

					- 是否跳过系统缓存。1 表示跳过系统缓存

				- iodepth

					- 表示使用异步 I/O（asynchronous I/O，简称 AIO）时，同时发出的 I/O 请求上限。

				- rw

					- 表示 I/O 模式。 read/write 分别表示顺序读 / 写，而 randread/randwrite 则分别表示随机读 / 写

				- ioengine

					- 表示 I/O 引擎，它支持同步（sync）、异步（libaio）、内存映射（mmap）、网络（net）等各种 I/O 引擎。

				- bs

					- 表示 I/O 的大小

				- filename

					- 表示文件路径，当然，它可以是磁盘路径（测试磁盘性能），也可以是文件路径（测试文件系统性能）

					- 注意，用磁盘路径测试写，会破坏这个磁盘中的文件系统，所以在使用前，你一定要事先做好数据备份。

			- 测试报告重点关注项

				- slat

					- 指从 I/O 提交到实际执行 I/O 的时长（Submission latency）

						- 对同步 I/O 来说，由于 I/O 提交和 I/O 完成是一个动作，所以 slat 实际上就是 I/O 完成的时间

				- clat

					- 指从 I/O 提交到 I/O 完成的时长（Completion latency）

						- 对同步 I/O 来说，clat 是 0

				- lat

					- 从 fio 创建 I/O 到 I/O 完成的总时长

				- bw

					- 代表吞吐量

				- iops

					- 每秒 I/O 的次数

		- I/O重放

			- 使用blktrace跟踪磁盘I/O，注意指定应用程序正在操作的磁盘

				- blktrace /dev/sdb

			- 将结果转化为二进制文件

				- blkparse sdb -d sdb.bin

			- 使用fio重放日志

				- fio --name=replay --filename=/dev/sdb --direct=1 --read_iolog=sdb.bin

### 进程

- 父子进程（线程）

	- 进程中包含的线程

		- pstree -g

			- 展示当前系统中正在运行的进程的树状结构（即进程下包含具体的线程）

	- 查找一个进程的父进程

		- pstree | grep xxx

- 进程状态

	- ps aux

- 批量kill 进程

	- ps aux | grep "/usr/bin/python2.7 app_0822.py" | awk {'print $2'}  | xargs kill

- 显示进程中的线程信息

	- ps -eLF

### 网络

- 出口IP

	- curl cip.cc

	- https://ip138.com/

- netstat

	- 显示tcp、udp链接的程序

		- netstat -pantu

	- 参数

		- -a 

			- 显示所有选项，默认不显示LISTEN相关

		- -t 

			- 仅显示tcp相关选项

		- -u

			- 仅显示udp相关选项

		- -n 

			- 拒绝显示别名，能显示数字的全部转化成数字

		- -l 

			- 仅列出有在 Listen (监听) 的服务状态

		- -p 

			- 显示建立相关链接的程序名

		- -r 

			- 显示路由信息，路由表

		- -e

			- 显示扩展信息，例如uid等

		- -s

			- 按各个协议进行统计

		- -c

			- 每隔一个固定时间，执行该netstat命令

### 环境

- 语言编码

	- export LC_ALL=zh_CN.UTF-8

### 文件

- 查看2个文件夹内哪些文件不同
diff -rq dir1 dir2

- 文件夹内递归计算文件md5


	- -type f：仅查找文件
-exec md5sum {} \;：对找到的每个文件执行 md5sum 命令
find /path/to/directory -type f -exec md5sum {} \;

	- -print0：输出文件名，以空字符（null character）分隔，处理文件名中包含空格或特殊字符的情况
xargs -0：从标准输入读取以空字符分隔的文件名，并传递给 md5sum 命令
find /path/to/directory  -type f -print0 | xargs -0 md5sum |sort

- 查询含有关键字的文件

	- cat *.h | grep --dereference-recursive "key" 

- 查询含有关键字的字符串，且请含有这个字符串，并去除

	- grep -hr "comm::ERR" . | grep -o 'comm::ERR[^;]*' |  sort |  uniq

- json文件中替换掉某个其中某个字段的image数据

	- 非数组

		- NEW_IMAGE_BASE64=$(base64 -w 0 test.jpeg)

		- jq --arg new_image "$NEW_IMAGE_BASE64" '.live_image.image = $new_image' input.json > output.json

	- 数组

		- 全替换

			- jq --arg new_image "$NEW_IMAGE_BASE64" '.live_image[].image = $new_image' input.json > output.json

		- 指定索引替换

			- jq --arg new_image "$NEW_IMAGE_BASE64" '.live_image[0].image = $new_image' input.json > output.json

### 调式

- go语言

	- [dlv](https://github.com/go-delve/delve/tree/master/Documentation/installation)

		- 安装

			- # 安装
go install github.com/go-delve/delve/cmd/dlv@latest

		- 使用

			- 在工程里启动 （-- 后面为启动参数）
dlv debug cmd/sealet/sealet.go -- --config ./conf/sealet.yaml

			- dlv exec binary

			- # 断点
break main.main
break main.go:40

			- 继续：c
下一行： n
函数内：s

	- [go调度调式](https://segmentfault.com/a/1190000020108079)

		- GODEBUG=schedtrace=1000  nohup ./sealet --config .conf/sealet.yaml 2>&1 &

	- [内存逃逸分析](https://geektutu.com/post/hpg-escape-analysis.html#:~:text=2.1-,%E4%BB%80%E4%B9%88%E6%98%AF%E9%80%83%E9%80%B8%E5%88%86%E6%9E%90,-%E5%9C%A8%20C%20%E8%AF%AD%E8%A8%80)

		- go build -gcflags="-m -m -l" ./test1.go

- 编译

	- 通过二进制查看依赖库

		- strings test.bin  | grep -i opencv

- c++

	- gdb

		- 显示所有线程堆栈

			- thread apply all bt

	- gcc

		- 在/etc/profile里增加，默认使用gcc7

			- source /opt/rh/devtoolset-7/enable

		- GCC编译器查找头文件和库文件的标准位置，以及用户自定义的位置

			- gcc -print-search-dirs

		- 查看gcc支持的cpp版本

			- # -x c++: 表示将输入文件视为C++源文件进行处理
# -std=c++14: 表示使用C++14标准进行编译
# -dM: 表示输出预定义宏的定义
# -E: 表示只进行预处理，不进行编译
# -: 表示从标准输入读取输入文件
g++ -x c++ -std=c++14 -dM -E - </dev/null|grep __cplusplus

	- c++filt

		- 显示函数原型c++filt 编译后函数名
c++filt _ZNSt6vectorISsSaISsEED1Ev

	- 编译、链接、运行信息

		-  undefined reference to `youtu::sdk::types::code::CgoError* google::protobuf::Arena::CreateMaybeMessage<youtu::sdk::types::code::CgoError>(google::protobuf::Arena*)'

			- 使用gcc7编译

		- 加入-g -O2 选项时编译，链接错误
undefined reference to 'xxx@GLIBCXX_3.4.21'

			- 编译时加参数--shared-libgcc

			- 在环境中加入gcc7(需要版本)的libstdc++.so

		- _GLIBCXX_USE_CXX11_ABI作用

			- 用于控制 libstdc++ 库的 C++11 ABI 版本，在 C++11 标准中，改变了一些 C++ 标准库的实现，因此在 C++11 标准之前和之后的程序之间可能存在 ABI 兼容性问题。

			- _GLIBCXX_USE_CXX11_ABI 的值为 0 时，使用旧的 ABI 版本，值为 1 时，使用新的 ABI 版本。通常情况下，如果您的程序是在 C++11 标准之后编译的，则应将其设置为 1，否则应将其设置为 0。

			- 检测so库_GLIBCXX_USE_CXX11_ABI是否为0

				- # 如果是c++11接口， nm能看到接口名字包含cxx11
nm libxx.so | grep cxx11 | c++filter

		- 编译信息查询
如：链接路径排查

			- cmake --trace

			- make VERBOSE=1

		- 收集工程下third_path下的库路径

			- COMPILE_THIRD_LIB_PATH=`find -L third_path/ -maxdepth 3 -mindepth 2 -type d \( -iname "lib*" -o -iname "stubs" \) -print0 |xargs -0 -I {} sh -c 'echo {}'|sed "s#^#$PWD/#g"|tr '\n' ':'`
export LD_LIBRARY_PATH=$COMPILE_THIRD_LIB_PATH:$LD_LIBRARY_PATH

			- #打包
for dep in `ldd "build/${target}" |grep "third_path"| awk '{if (match($3,"/")){ print $3 }}' | grep -v "^/lib64"`
do
  echo $dep
  cp -v -L $dep ${packdir}/lib
done

		- 查看程序的rpath路径

			- readelf -a build/bin/test 2>&1  | grep -i rpath

		- 一个进程中test，liba.so 静态链接了openssl 1.0.1版本，服务框架静态链接了openssl 1.0.2版本，导致服务crash

			- 确认openssl版本

				- strings liba.so | grep -i openssl

				- strings test | grep -i openssl

			- crash原因

				- 比如liba.so中调用了自己openssl库的 c接口，然后c接口又调用了服务框架内openssl中的d接口

				- 查看调用依赖

					- nm liba.so | grep -i openssl
T 可能调用外面的库接口（test静态链接）
t 调用自己的库接口

		- 不同cuda驱动版本

			- 当前是cuda11，需要同时支持cuda11，cuda12，但是不支持cuda10

				- if [ -f /home/data/model/lib64/libcuda.so ]; then rm -f /home/data/model/lib64/libcuda.so*; fi
if [ -f /home/data/model/lib64/libnvrtc-builtins.so.11.0 ]; then rm -f /home/data/model/lib64/libnvrtc-builtins.so.11.0*; fi
if [ -f /home/data/model/lib64/libnvidia-ptxjitcompiler.so ]; then rm -f /home/data/model/lib64/libnvidia-ptxjitcompiler.so*; fi

- pporf

	- 安装

		- svg图，依赖graphviz

			- mac

				- brew install graphviz

			- centos

				- yum install graphviz

		- 火焰图，依赖go-torch

			- go get github.com/uber/go-torch

			- git clone git@github.com:brendangregg/FlameGraph.git

			- export PATH=$PATH:/path/to/FlameGraph

	- 使用

		- 使用github.com/pkg/profile包进行性能数据采集

		- 将pprof性能数据文件生成svg文件

			- go tool pprof -svg  cpu.pprof > cpu.svg

		- 将pprof性能数据文件生成pdf文件

			- go tool pprof -pdf  cpu.pprof > cpu.pdf

			- curl -X GET  http://localhost:10001/debug/pprof/profile --output cpu.pprof

			- # 内存分析，如果这种方法无法看出go的内存泄漏问题，可能是有cgo导致的
# cgo的排查方法需要参考cpp程序的tcmalloc的使用，比较前后2次的内存差值
curl -X GET  http://localhost:10001/debug/pprof/heap--output mem.pprof

		- 将pprof性能数据文件生成火焰图

			- go-torch --binaryinput=cpu.pprof

		- web界面分析（如cpu分析）

			- # 更方便
go tool pprof -http=:9999 cpu.pprof

				- 相关字段含义

					- flat

						- 本函数的执行耗时

					- flat%

						- flat 占 CPU 总时间的比例。程序总耗时 16.22s, Eat 的 16.19s 占了 99.82%

					- sum%

						- 前面每一行的 flat 占比总和

					- cum

						- 累计量。指该函数加上该函数调用的函数总耗时

					- cum%

						- cum 占 CPU 总时间的比例

- 系统调式

	- coredump

		- 设置coredump路径

			- 在/proc/sys/kernel/core_pattern文件里设置core路径，如cat /proc/sys/kernel/core_pattern：
/data/timatrix/coredump/core-%e-%p-%t-%s

		- 生成coredump

			- ulimit -c unlimited

		- c++程序增加coredump

			- ulimit -c unlimited; echo '/app/coredump/core-%e-%p-%t' > /proc/sys/kernel/core_pattern; ./bin/test 

		- go程序增加coredump

			- # 程序启动命令行加上 env GOTRACEBACK=crash 
nohup env GOTRACEBACK=crash ./test
-config_path=../conf/app.conf -consul_server="192.168.0.110:8500"
-data_receive_url="http://cgi.service.consul:10010/youmall/capture"
-monitor_config=../conf/monitor_config.yaml -benchmark=false >> error.log &

	- 清除cache

		- echo 3 > /proc/sys/vm/drop_caches; sync

	- 在有共享内存的情况下统计服务内存

		- 每次测试前，需要把服务先停了

		- 执行ipcrm -a命令

		- 启动服务

		- 测试，用top命令看，res已经包含了共享内存

	- 查看系统日志

		- dmesg -T > sys.log

- 音视频

	- live555

		- 点播服务器

			- 安装

				- wget  http://www.live555.com/liveMedia/public/live555-latest.tar.gz

				- ./genMakefiles linux-64bit

				- make install

			- 使用

				- 在live555MediaServer同目录下新建一个文件夹来存放视频， 如test_video/test.mp4

				- # 文件的rtsp地址为：
rtsp://127.0.0.1/test_video/test.mp4

	- ffmpeg

		- 获取视频总帧数

			- ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0   /data/test.ts

		- 图片编码

			- 保存一张实时流图片

				- ffmpeg -i "rtsp://admin:admin@127.0.0.1:554/cam/realmonitor?channel=1&subtype=0" -y -f mjpeg -t 0.001 test.jpg

		- 视频编码

			- 转换到yuv420格式

				- # -an编码去掉音频
ffmpeg -i test1080p.ts -c:v libx264 -an -pix_fmt yuv420p test1080p.crowd.mute.ts

			- 限制编码质量crf和gop

				- ffmpeg -i cosplay.mp4 -vcodec libx264  -preset fast -crf ${crf} -g ${gop} -sc_threshold 0 -y -movflags faststart crf_${crf}_gop_${gop}.mp4

			- 限制比特率

				- ffmpeg -i cosplay.mp4 -c:v libx264 -preset fast  -b:v ${bitrate}  output.{bitrate}.mp4

			- 限制编码质量的前提下，进一步限制码率

				- ffmpeg -i cosplay.mp4 -c:v libx264 -preset fast -crf ${crf} -g ${gop} -b:v ${bitrate} -maxrate ${bitrate}  -bufsize 50k -sc_threshold 0 -y -movflags faststart crf_${crf}_gop_${gop}_bitrate_${bitrate}.mp4

		- [命令行](https://github.com/xufuji456/FFmpegAndroid/blob/master/doc/FFmpeg_command_line.md)

- 动/静态库

	- [nm 命令查看函数符号表](https://stackoverflow.com/questions/37531846/nm-symbol-output-t-vs-t-in-a-shared-so-library#:~:text=T%2Ft%20means%20the%20symbol,are%20having%20upper%20case%20T.)

		- [ ](https://blog.csdn.net/xuq09/article/details/87970376)

			- T全局

			- t局部

			- U在其他定义定义

	- -Lxxx 链接库时，报找不到该库

		- 服务链接liba.so, 而liba.so又依赖了libb.so, 这个时候要保证-L链接的liba.so和libb.so的版本要完全一致

		- 通过bazel里BUILD文件方式来链接，因为内部通过了编译为新的name，所以会忽略掉soname不一致找不到库的问题，但是如果部署的时候，用不同的版本，会crash

		- readelf -d liba.so 可以查看soname

### 测试

- http

	- multipart/form-data

		- curl -X POST http://ti-data-center-backend.ti-base:11000/UploadFiles -F "upload=@/data/ti-platform-fs/tidatacenter/test.mp4"  -F "filename=/data/ti-platform-fs/tidatacenter/datacenter/0319/13557026373776187393/test.mp4" -H "Content-Type: multipart/form-data"

		- 带登录态信息

			- time curl -sSL -X POST -D - http://172.16.0.99/gateway -F "upload=@/data/ti-platform-fs/tidatacenter/5g.mp4"  -F "filename=/data/ti-platform-fs/tidatacenter/datacenter/0319/13557026373776187397/test.mp4" -H "Content-Type: multipart/form-data" -H 'X-TC-Action: UploadFiles' -H 'X-TC-Version: 2021-03-17' -H 'Cookie: ti-userid=10001; alauda_locale=zh; ti-ticket=KTL713EqCMJ2UWRH0yDXk0LyVo1XM5nfnQLpr5+Nl30g7G2arSrLOj+ZLBI/m4U=; ti-token=491e05a9-98d4-43cf-b991-2ba43bb2fe3d; ti-user=root' -H 'X-TC-Titoken: 491e05a9-98d4-43cf-b991-2ba43bb2fe3d' -o /dev/null

	- put文件

		- 子主题 1

	- curl请求

		- curl --request POST --url http://127.0.0.1:12102/api/GetConfig -d "@./req.json"

		- curl -X POST -H "Content-Type: application/json" -d  @./DetectColor.json http://127.0.0.1:12102/DetectColor

			- e7c55a1f-79c6-4103-9de1-b547f8590ffc

		- # -d 参数进行了url编码， --data-binary是二进制文件
curl -X  POST -H "Content-Type: application/x-protobuf" --data-binary    "@./testdata/search.pb" http://localhost:15025/youtu.image.ImageSearchService/SearchImage --output -

		- # 读取文件，进行base64
# -d@- 从标准输出中读取内容
 echo -n '{"session_id": "kay", "image_a": "'"$(base64  -w 0 ./blackfeature/black4.jpg)"'", "image_b": "'"$(base64 -w 0 ./blackfeature/black4.jpg)"'"}' | curl -H "Content-Type: application/json" -d @-  http://127.0.0.1:31739/CompareFace

	- 代理

		- TCP篡改

			- curl -vvv -o /dev/null -H "HOST: quyujiaofu-new-1300074211.cos.ap-guangzhou.myqcloud.com" http://9.139.9.174:8080/hk_test/male_1080_1440p.jpg

			- curl --proxy "https://9.139.9.174:8080" http://quyujiaofu-new-1300074211.cos.ap-guangzhou.myqcloud.com/hk_test/male_1080_1440p.jpg  -o /dev/null

		- 代理方式

			- curl --proxy "http://9.139.9.174:8080" https://quyujiaofu-new-1300074211.cos.ap-guangzhou.myqcloud.com/hk_test/zujunchen/male_1080_1440p.jpg  -o /dev/null

- 压测

	- http

		- ab -n 1000 -c 100 -T application/json -p ./req.json "http://127.0.0.1:11000/DescribeUploadUrl"

		- [gohttpbench  -n 500 -c 100 -T application/json -p ./testdata/req.json http://127.0.0.1:30010/ReportOccupy](https://github.com/parkghost/gohttpbench)

		- [# 包含吞吐量
./bombardier   -n 30000 -c 3000  -t 10s -H Content-type:application/json   -f ./testdata/req.480p.json -m POST -l  http://127.0.0.1:30010/ReportOccupy](https://github.com/codesenberg/bombardier)

	- grpc

		- [ghz --insecure  --proto api/openapi-spec/sealet/date/v1/date.proto --call sea.api.sealet.date.v1.DateService/Now -c 100 -n 10000 -d '{"name":"Joe"}'  127.0.0.1:10001](https://ghz.sh/docs/examples)

		- 使用配置文件测试

			-  ghz --insecure --debug=./ghz.log  --proto ./test/traffic_recognition_service.proto --call youtu.traffic.trafficapi/RecVehicle  -B ./recVehicleReq.pb.binary  -c 1 -n 1  127.0.0.1:10098

- 抓包

	- tcpdump

		- 参数

			- -n  

				- 不进行IP地址到主机名的转换

			- -A

				- 以ascii的方式显示数据包，抓取web数据时很有用

			- -c 

				- 在收到指定数目的包后，tcpdump就会停止

			- -s

				- 抓取数据包时默认抓取长度为68字节。加上 -s 0 后可以抓到完整的数据包

			- -i

				- interface 监听的网卡

			- -vv 

				- 输出详细的报文信息

		- tcpdump -i eth0 -vvvs 1024 -l -n -A  tcp port 14000 > tcpcap.txt

		- # -w file 导出的文件，可用wireshark工具打开分析
tcpdump -i eth0 -vvvs 1024 -l -n -A  tcp port 14000 -w tcpcap.txt

