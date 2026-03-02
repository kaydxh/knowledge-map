# kubernetes

## docker 

### 核心功能

- 就是通过约束和修改进程的动态表现（计算机里的数据和状态的总和），从而为其创造出一个“边界”

	- 进程视图

		- Namespace 

			- 主要特点

				- 容器，其实是一种特殊的进程而已。只需要将该进程的信息与其它进程镜像隔离

				- 通过 linux的系统调用clone功能实现进程namepsace隔离。Linux 创建新进程的一个可选参数。在 Linux 系统中创建现场的系统调用是 clone()，linux的线程是通过进程实现的

				  int pid = clone(main_function, stack_size, SIGCHLD, NULL); 
				  
				  # 参数中指定 CLONE_NEWPID 参数， 一个全新的进程空间，在这个进程空间里它的pid为1
				  int pid = clone(main_function, stack_size, CLONE_NEWPID | SIGCHLD, NULL);
				- Linux 操作系统还提供了PID、 Mount、UTS、IPC、Network 和 User 这些 Namespace，用来对各种不同的进程上下文进行“障眼法”操作，只让被隔离的进程看到当前namespace里信息

	- 资源约束

		- linux Cgroups (Linux Control Group)

			- Linux Cgroups的设计其实就是一个子系统目录加上一组资源限制文件的组合

				- 主要功能

					- 限制一个进程组能够使用的资源上限，包括 CPU、内存、磁盘、网络带宽等等

				- 暴露的接口是文件系统

					- 展示已挂载cgroups的文件系统（即可以被cgroups限值的资源种类），如果没有目录输出，就需要执行挂载

						- mount -t cgroup 

					- 限制cpu资源

						- 在/sys/fs/cgroup/cpu目录下，创建自己的一个文件夹，如container，操作系统自动在该目录下生成资源限制文件

							- mkdir container

								- # quota 还没有任何限制（即：-1）
# CPU period 则是默认的 100 ms（100000 us）
# 即每100ms时间内，该控制限制组的进程可已使用100%，如需限制到cpu 20%的资源，在cfs_quota文件里写入20000（即20ms）
echo 20000 > /sys/fs/cgroup/cpu/container/cpu.cfs_quota_us

# 将被限制的进程pid(如，226)写入到continer组里的tasks文件，上面修改的cpu限制就会生效了
echo 226 > /sys/fs/cgroup/cpu/container/tasks

					- 其它资源

						- blkio，为块设备设定I/O 限制，一般用于磁盘等设备

						- cpuset，为进程分配单独的 CPU 核和对应的内存节点；memory，为进程设定内存使用的限制

						- memory，为进程设定内存使用的限制

			- 对于linux 容器来说，只需要在每个子系统下面创建一个容器组（即一个新的目录，一般为容器id最为目录），然后在启动进程的时候把这个进程的pid写入到控制组的tasks文件中

			- 具体的资源限制值，可以通过docker run启动参数传入

				- docker run -it --cpu-period=100000 --cpu-quota=20000 ubuntu /bin/bash

### 与虚拟机的区别

- 虚拟机

	- 优势

		- 不依赖宿主机内核，可以在window系统的宿主机上虚拟化linux服务器

	- 劣势

		- 虚拟化技术作为应用沙盒，必须有Hypervisor来创建虚拟机，这个虚拟机是真实存在的，且必须运行一个完整的Guest OS才能执行用户的应用进程。因此资源成本比较大

		- 一个运行centos的KVM虚拟机启动后，虚拟机自身需要占用100～300MB内存

		- 用户应用在虚拟机里系统调用时，需要经过虚拟机的拦截和处理，这本身也是一种消耗，尤其对计算资源、网络和磁盘 I/O 的损耗非常大

- 容器

	- 优势

		- 容器化后的应用进程依然是宿主机上的一个普通应用，因此不存在虚拟化带来的消耗

		- Namespace作为隔离容器的手段，不需要单独的Guest OS

	- 劣势

		- 容器只是运行在宿主机上一种特殊的进程，多个容器之间使用的就还是同一个宿主机的操作系统内核

		  这就是在windows宿主机上运行linux容器是不可能的，或者在低版本的linux宿主机上运行高版本的linux容器也是不可能的
		- 在linux内核中，有很多资源和对象是不能被Namespace化的，最典型的就是时间

			- 如果在容器里使用settimeofday(2)系统调用修改时间，整个宿主机的时间就会被随之更改

		- 容器共享了宿主机的内核，因此安全度相比虚拟机就比较低

			- 实际可以使用Seccomp过滤调部分系统调用，但是会消耗性能，同时无法预先精确知道哪些系统调用应该被过滤

### 单进程模型

- 用户的应用进程在容器里的pid就是1，也就是后续创建其它进程的父进程

	- 容器里只能启动一个应用服务，如果启动多个应用程序，只要pid为1的进程没有crash，就不会重启容器

- 可以使用systemd 或者 supervisord 这样的软件来代替应用本身作为容器的启动进程（不是最佳方法）

### rootfs(根文件系统)

- 特点

	- 挂载在容器根目录上、用来为容器进程提供隔离后执行环境的文件系统

	- rootfs 只是一个操作系统所包含的文件、配置和目录，并不包括操作系统内核。在 Linux 操作系统中，这两部分是分开存放的，操作系统只有在开机启动时才会加载指定版本的内核镜像

		- 决定了

			- rootfs保证了环境的一致性

	- 同一台机器上的所有容器，都共享宿主机操作系统的内核

		- 决定了

			- 应用程序需要配置内核参数、加载额外的内核模块，以及跟内核进行直接的交互，就需要注意了：这些操作和依赖的对象，都是宿主机操作系统的内核，它对于该机器上的所有容器来说是一个“全局变量”，牵一发而动全身

- 镜像的层

	- 命令查看

		-  docker image inspect ubuntu:latest

			- 222.png

	- 特点

		- 这些层就是增量 rootfs，每一层都是 Centos操作系统文件与目录的一部分；而在使用镜像时，Docker 会把这些增量联合挂载在一个统一的挂载点上（等价于前面例子里的“/C”目录. 这个挂载点就是

			- 统一挂载点

				- centos

					- /var/lib/docker/overlay2/

	- 组成

		- 只读层

			- 它是这个容器的 rootfs 最下面的五层，对应的正是 ubuntu:latest 镜像的五层。可以看到，它们的挂载方式都是只读的（ro+wh，即 readonly+whiteout)

		-  可读写层

			- 它是这个容器的 rootfs 最上面的一层（6e3be5d2ecccae7cc），它的挂载方式为：rw，即 read write。在没有写入文件之前，这个目录是空的。而一旦在容器里做了写操作，你修改产生的内容就会以增量的方式出现在这个层中

		- Init 层

			- 它是一个以“-init”结尾的层，夹在只读层和读写层之间。Init 层是 Docker 项目单独生成的一个内部层，专门用来存放 /etc/hosts、/etc/resolv.conf 等信息

### 核心原理实际上就是为待创建的用户进程做如下操作

- 启用 Linux Namespace 配置

- 设置指定的 Cgroups 参数

- 切换进程的根目录（Change Root）。（优先使用 pivot_root 系统调用，如果系统不支持，才会使用 chroot

### 基本命令

- docker tag 命令给容器镜像起一个完整的名字

	- docker tag helloworld geektime/helloworld:v1

- 将镜像上传到 Docker Hub

	- docker push geektime/helloworld:v1

- 将一个正在运行的容器，提交位一个镜像

	- docker commit 4ddf4638572d geektime/helloworld:v2

		- docker commit，实际上就是在容器运行起来后，把最上层的“可读写层”，加上原先容器镜像的只读层，打包组成了一个新的镜像。当然，下面这些只读层在宿主机上是共享的，不会占用额外的空间

		- 由于使用了联合文件系统，你在容器里对镜像 rootfs 所做的任何修改，都会被操作系统先复制到这个可读写层，然后再修改。这就是所谓的：Copy-on-Write

- 查看当前正在运行容器的进程号

	- docker inspect --format '{{ .State.Pid }}'  4ddf4638572d

- 查看指定进程的所有 Namespace 对应的文件

	-  ls -l  /proc/25686/ns

		- 这就意味着：一个进程，可以选择加入到某个进程已有的 Namespace 当中，从而达到“进入”这个进程所在容器的目的，这正是 docker exec 的实现原理

			- 2wx.png

- 挂载目录

	- 命令

		- # 由于没有显示声明宿主机目录，那么 Docker 就会默认在宿主机上创建一个临时目录 /var/lib/docker/volumes/[VOLUME_ID]/_data，然后把它挂载到容器的 /test 目录上
docker run -v /test ...

		- # 把宿主机的 /home 目录挂载到容器的 /test 目录上
docker run -v /home:/test ...

	- 原理

		- 这里使用到的挂载技术，就是 Linux 的绑定挂载（bind mount）机制

		- 绑定挂载实际上是一个 inode 替换的过程。在 Linux 操作系统中，inode 可以理解为存放文件内容的“对象”，而 dentry，也叫目录项，就是访问这个 inode 所使用的“指针”

		- mount --bind /home /test，会将 /home 挂载到 /test 上。其实相当于将 /test 的 dentry，重定向到了 /home 的 inode。这样当我们修改 /test 目录时，实际修改的是 /home 目录的 inode。这也就是为何，一旦执行 umount 命令，/test 目录原先的内容就会恢复：因为修改真正发生在的，是 /home 目录里

			- 3wx.png

### 全景图

- 图片

	-  

		- 5wx.jpg

- 含义

	- rootfs 层的最下层，是来自 Docker 镜像的只读层

	- 在只读层之上，是 Docker 自己添加的 Init 层，用来存放被临时修改过的 /etc/hosts 等文件

	- 而 rootfs 的最上层是一个可读写层，它以 Copy-on-Write 的方式存放任何对只读层的修改，容器声明的 Volume 的挂载点，也出现在这一层

## kubernetes架构

### 架构图

-  

### 流程

-  

	- [ ](https://zhuanlan.zhihu.com/p/382229383)

- 扩容步骤例子

	- 通过kubectl命令行工具向API Server发送一个请求：创建ReplicaSet，API Server会将此请求存储在etcd中

	- Controller Manager会接受到一个通知

	- Controller Manager发现现在的集群状态和预期的状态不一致，因此需要创建Pod，此信息会通知到Scheduler

	- Scheduler会选择空闲的Worker节点，然后通过API Server更新Pod的定义

	- API Server会通知到Worker节点的上的kubelet

	- kubelet指示当前节点上的Container Runtime运行对应的容器

	- Container Runtime下载镜像并启动容器

### 架构

- 控制节点

	- 作用

		- 由三个紧密协作的独立组件组合而成，它们分别是负责 API 服务的 kube-apiserver、负责调度的 kube-scheduler，以及负责容器编排的 kube-controller-manager

	- 组成

		- Master

			- API Server

				- 负责 API 服务

				- 依赖

					- 组件

						- Etcd

							- 整个集群的持久化数据，则由 kube-apiserver 处理后保存在 Etcd 中

					- 通信方式

						- grpc

			- Controller Manager

				- 负责容器编排

			- Scheduler

				- 负责调度

- 计算节点

	- 作用

		- kubelet 主要负责同容器运行时（比如 Docker 项目）打交道。而这个交互所依赖的，是一个称作 CRI（Container Runtime Interface）的远程调用接口，这个接口定义了容器运行时的各项核心操作

	- 组成

		- kubelet

			- Runtime(容器运行时)

				- CRI（Container Runtime Interface）容器运行时接口

			- Networking(容器网络)

				- CNI（Container Networking Interface）容器网络接口

			- Storage(容器存储)

				- CSI（Container Storage Interface）容器存储接口

			- Device（容器设备）

				-  gRPC 协议

				- 作用

					- 管理 GPU 等宿主机物理设备的主要组件，也是基于 Kubernetes 项目进行机器学习训练、高性能作业支持等工作必须关注的功能

## kubernetes Workloads(作业管理)

### pod

- 定义

	- 是 Kubernetes 项目中最小的 API 对象

	- Pod 扮演的是传统部署环境里“虚拟机”的角色

- 作用

	- 以pod为单位，去调度节点，避免出现成组调度没有妥善处理问题（类似进程和进程组的关系）

		- 例子

			- 有3个程序，A， B， C，都需要1G的内存，且这3个容器具有亲和性，必须被调度在同一个node上，假如有node-1（3G可用）， node-2（2GB可用），如果没有pod，那么可能出现A、B被调度在node-2节点，而C因为亲和性必须被调度在node-2，但是node-2没有可用内存。如果用了pod（Pod 是 Kubernetes 里的原子调度单位），那么就会去找内存>=3G的node，（Kubernetes 项目的调度器，是统一按照 Pod 而非容器的资源需求进行计算的）

- 原理

	- pod只是个逻辑概念，Kubernetes 真正处理的，还是宿主机操作系统上 Linux 容器的 Namespace 和 Cgroups，而并不存在一个所谓的 Pod 的边界或者隔离环境，Pod，其实是一组共享了某些资源的容器

	- pod的实现需要使用一个中间Infra容器，Infra 容器永远都是第一个被创建的容器，而其他用户定义的容器，则通过 Join Network Namespace 的方式，与 Infra 容器关联在一起。这样的组织关系，可以用下面这样一个示意图来表达 

		-  

- 容器设计模式

	- sidecar(组合”操作）

		- 定义

			- 可以在一个 Pod 中，启动一个辅助容器，来完成一些独立于主进程（主容器）之外的工作

		- 例子

			- WAR 包与 Web 服务器

				- apiVersion: v1
kind: Pod
metadata:
  name: javaweb-2
spec:
  initContainers:
  - image: geektime/sample:v2
    name: war
    command: ["cp", "/sample.war", "/app"]
    volumeMounts:
    - mountPath: /app
      name: app-volume
  containers:
  - image: geektime/tomcat:7.0
    name: tomcat
    command: ["sh","-c","/root/apache-tomcat-7.0.42-v2/bin/start.sh"]
    volumeMounts:
    - mountPath: /root/apache-tomcat-7.0.42-v2/webapps
      name: app-volume
    ports:
    - containerPort: 8080
      hostPort: 8001 
  volumes:
  - name: app-volume
    emptyDir: {}

					- 在 Pod 中，所有 Init Container 定义的容器，都会比 spec.containers 定义的用户容器先启动。并且，Init Container 容器会按顺序逐一启动，而直到它们都启动并且退出了，用户容器才会启动

			- 容器的日志收集

				- 有一个应用，需要不断地把日志文件输出到容器的 /var/log 目录中

					- 做法

						- 把一个 Pod 里的 Volume 挂载到应用容器的 /var/log 目录上

						- 在这个 Pod 里同时运行一个 sidecar 容器，它也声明挂载同一个 Volume 到自己的 /var/log 目录上

						- 接下来 sidecar 容器就只需要做一件事儿，那就是不断地从自己的 /var/log 目录里读取日志文件，转发到 MongoDB 或者 Elasticsearch 中存储起来。这样，一个最基本的日志收集工作就完成

						- 注：Istio 项目使用 sidecar 容器完成微服务治理的原理

- pod级别

	- pod级别属性

		-  凡是调度、网络、存储，以及安全相关的属性

			- 配置这个“机器”的网卡（即：Pod 的网络定义）

			- 配置这个“机器”的磁盘（即：Pod 的存储定义）

			- 配置这个“机器”的防火墙（即：Pod 的安全定义）

			- 这台“机器”运行在哪个服务器之上（即：Pod 的调度）

		- 凡是跟容器的 Linux Namespace 相关的属性

			- shareProcessNamespace=true ,这就意味着这个 Pod 里的容器要共享 PID Namespace

				- apiVersion: v1
kind: Pod
metadata:
  name: nginx
spec:
  shareProcessNamespace: true
  containers:
  - name: nginx
    image: nginx
  - name: shell
    image: busybox
    stdin: true
    tty: true

		- 凡是 Pod 中的容器要共享宿主机的 Namespace

			- 共享宿主机的 Network、IPC 和 PID Namespace。这就意味着，这个 Pod 里的所有容器，会直接使用宿主机的网络、直接与宿主机进行 IPC 通信、看到宿主机里正在运行的所有进程

				- apiVersion: v1
kind: Pod
metadata:
  name: nginx
spec:
  hostNetwork: true
  hostIPC: true
  hostPID: true
  containers:
  - name: nginx
    image: nginx
  - name: shell
    image: busybox
    stdin: true
    tty: true

	- pod级别字段

		- NodeSelector

			- 提供用户将 Pod 与 Node 进行绑定的字段

				- # 这个 Pod 永远只能运行在携带了“disktype: ssd”标签（Label）的节点上；否则，它将调度失败
apiVersion: v1
kind: Pod
...
spec:
 nodeSelector:
   disktype: ssd

		- NodeName

			- 一旦 Pod 的这个字段被赋值，Kubernetes 项目就会被认为这个 Pod 已经经过了调度，调度的结果就是赋值的节点名字。所以，这个字段一般由调度器负责设置，但用户也可以设置它来“骗过”调度器，当然这个做法一般是在测试或者调试的时候才会用到

		- HostAliases

			- 定义了 Pod 的 hosts 文件（比如 /etc/hosts）里的内容
不能直接修改hosts文件内容，在 Pod 被删除重建之后，kubelet 会自动覆盖掉被修改的内容

				- apiVersion: v1
kind: Pod
...
spec:
  hostAliases:
  - ip: "10.1.2.3"
    hostnames:
    - "foo.remote"
    - "bar.remote"

		- shareProcessNamespace

			- shareProcessNamespace=true ,这就意味着这个 Pod 里的容器要共享 PID Namespace

		- spec部分

			-  Containers 

				- Image（镜像）、Command（启动命令）、workingDir（容器的工作目录）、Ports（容器要开发的端口），以及 volumeMounts（容器要挂载的 Volume）都是构成 Kubernetes 项目中 Container 的主要字段

				- 子字段

					- ImagePullPolicy

						- 定义了镜像拉取的策略

							- 含义

								- ImagePullPolicy 的值默认是 Always，即每次创建 Pod 都重新拉取一次镜像

								- 当容器的镜像是类似于 nginx 或者 nginx:latest 这样的名字时，ImagePullPolicy 也会被认为 Always

								- 而如果它的值被定义为 Never 或者 IfNotPresent，则意味着 Pod 永远不会主动拉取这个镜像，或者只在宿主机上不存在这个镜像时才拉取

					- Lifecycle 字段

						- 它定义的是 Container Lifecycle Hooks。顾名思义，Container Lifecycle Hooks 的作用，是在容器状态发生变化时触发一系列“钩子”

							- 例子

								- apiVersion: v1
kind: Pod
metadata:
  name: lifecycle-demo
spec:
  containers:
  - name: lifecycle-demo-container
    image: nginx
    lifecycle:
      postStart:
        exec:
          command: ["/bin/sh", "-c", "echo Hello from the postStart handler > /usr/share/message"]
      preStop:
        exec:
          command: ["/usr/sbin/nginx","-s","quit"]

							- 含义

								- postStar

									- 它指的是，在容器启动后，立刻执行一个指定的操作。postStart 并不严格保证顺。也就是说，在 postStart 启动时，ENTRYPOINT 有可能还没有结束

								- preStop 

									- 发生的时机，则是容器被杀死之前（比如，收到了 SIGKILL 信号）。preStop 操作的执行，是同步的(它会阻塞当前的容器杀死流程，直到这个 Hook 定义操作完成之后，才允许容器被杀死)

					- livenessProbe

						- 健康检查“探针”

							- kubelet 就会根据这个 Probe 的返回值决定这个容器的状态，而不是直接以容器镜像是否运行（来自 Docker 返回的信息）作为依据。这种机制，是生产环境中保证应用健康存活的重要手段

								- 方式

									- 命令请求

										- 
apiVersion: v1
kind: Pod
metadata:
  labels:
    test: liveness
  name: test-liveness-exec
spec:
  containers:
  - name: liveness
    image: busybox
    args:
    - /bin/sh
    - -c
    - touch /tmp/healthy; sleep 30; rm -rf /tmp/healthy; sleep 600
    livenessProbe:
      exec:
        command:
        - cat
        - /tmp/healthy
      initialDelaySeconds: 5
      periodSeconds: 5

									- http请求

										- 
...
livenessProbe:
     httpGet:
       path: /healthz
       port: 8080
       httpHeaders:
       - name: X-Custom-Header
         value: Awesome
       initialDelaySeconds: 3
       periodSeconds: 3

									- tcp请求

										- 
    ...
    livenessProbe:
      tcpSocket:
        port: 8080
      initialDelaySeconds: 15
      periodSeconds: 20

			- restartPolicy

				- 定义值

					- Always

						- 在任何情况下，只要容器不在运行状态，就自动重启容器

					- OnFailure

						- 只在容器 异常时才自动重启容器

					- Never

						- 从来不重启容器

				- 与Pod 状态的对应关系

					- 只要 Pod 的 restartPolicy 指定的策略允许重启异常的容器（比如：Always），那么这个 Pod 就会保持 Running 状态，并进行容器重启。否则，Pod 就会进入 Failed 状态

					- 对于包含多个容器的 Pod，只有它里面所有的容器都进入异常状态后，Pod 才会进入 Failed 状态。在此之前，Pod 都是 Running 状态。此时，Pod 的 READY 字段会显示正常容器的个数

	- pod生命周期

		- 定义

			- Pod 生命周期的变化，主要体现在 Pod API 对象的 Status 部分

		- 状态

			- Status

				- Pending

					- 这个状态意味着，Pod 的 YAML 文件已经提交给了 Kubernetes，API 对象已经被创建并保存在 Etcd 当中。但是，这个 Pod 里有些容器因为某种原因而不能被顺利创建。比如，调度不成功

				- Running

					- 这个状态下，Pod 已经调度成功，跟一个具体的节点绑定。它包含的容器都已经创建成功，并且至少有一个正在运行中

				- Succeeded

					- 这个状态意味着，Pod 里的所有容器都正常运行完毕，并且已经退出了。这种情况在运行一次性任务时最为常见

				- Failed

					- 这个状态下，Pod 里至少有一个容器以不正常的状态（非 0 的返回码）退出。这个状态的出现，意味着你得想办法 Debug 这个容器的应用，比如查看 Pod 的 Events 和日志

				- Unknown

					- 这是一个异常状态，意味着 Pod 的状态不能持续地被 kubelet 汇报给 kube-apiserver，这很有可能是主从节点（Master 和 Kubelet）间的通信出现了问题

			- Conditions

				- 作用

					- 用于描述造成当前 Status 的具体原因是什么

				- 细分字段

					- PodScheduled

						- 它的调度出现了问题

					- Ready

						- 已经可以对外提供服务

					- Initialized

					- Unschedulable

	- 使用阶段

### replicaSet(容器副本)

- 功能

	- 更新了 Deployment 的 Pod 模板（比如，修改了容器的镜像），那么 Deployment 就需要遵循一种叫作“滚动更新”（rolling update）的方式，来升级现有的容器

		- 将一个集群中正在运行的多个 Pod 版本，交替地逐一升级的过程，就是“滚动更新

- 与Deploymenet关系

	- 一个 ReplicaSet 对象，其实就是由副本数目的定义和一个 Pod 模板组成的。它的定义其实是 Deployment 的一个子集

	- Deployment 控制器实际操纵的，是ReplicaSet 对象，而不是 Pod 对象

	- 对于一个 Deployment 所管理的 Pod，它的 ownerReference 是ReplicaSet

	-  

		- 附件

			- k13.jpg

- 命令

	- 水平扩展/收缩

		- kubectl scale deployment nginx-deployment --replicas=4

### Deployment

- 实现原理(控制循环)

	- 1. Deployment 控制器从 Etcd 中获取到所有携带了“app: nginx”标签的 Pod，然后统计它们的数量，这就是实际状态；

	- 2. Deployment 对象的 Replicas 字段的值就是期望状态；

	- 3. Deployment 控制器将两个状态做比较，然后根据比较结果，确定是创建 Pod，还是删除已有的 Pod

- 字段

	- 附件

		- k11.png

	- 控制器定义

		- replicas

			- 定义被管理对象的期望状态

	- 被控制对象

		- pod模版

			- template

				- 跟一个标准的 Pod 对象的 API 定义一样

				- 所有被这个 Deployment 管理的 Pod 实例，其实都是根据这个 template 字段的内容创建出来的

- 命令

	- 创建deployment

		- # record 参数。它的作用，是记录下你每次操作所执行的命令，以方便后面查看
kubectl create -f nginx-deployment.yaml --record

	- 检查deployment 创建后的状态信息

		- 
$ kubectl get deployments
NAME               DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
nginx-deployment   3         0         0            0           1s

			- DESIRED：用户期望的 Pod 副本个数（spec.replicas 的值）

			- CURRENT：当前处于 Running 状态的 Pod 的个数

			- UP-TO-DATE：当前处于最新版本的 Pod 的个数，所谓最新版本指的是 Pod 的 Spec 部分与 Deployment 里 Pod 模板里定义的完全一致

			- AVAILABLE：当前已经可用的 Pod 的个数，即：既是 Running 状态，又是最新版本，并且已经处于 Ready（健康检查正确）状态的 Pod 的个数

	- 实时查看 Deployment 对象的状态变化

		- 
$ kubectl rollout status deployment/nginx-deployment
Waiting for rollout to finish: 2 out of 3 new replicas have been updated...
deployment.apps/nginx-deployment successfully rolled out

	- 查看一下这个 Deployment 所控制的 ReplicaSet

		- 
$ kubectl get rs
NAME                          DESIRED   CURRENT   READY   AGE
nginx-deployment-3167673210   3         3         3       20s

	- 修改 Deployment 

		- kubectl edit deployment/nginx-deployment

	- 查看 Deployment 的 Events

		- $ kubectl describe deployment nginx-deployment

	- 直接修改 deployment 所使用的镜像

		- $ kubectl set image deployment/nginx-deployment nginx=nginx:1.91
deployment.extensions/nginx-deployment image updated

	- 查看每次 Deployment 变更对应的版本

		- 
$ kubectl rollout history deployment/nginx-deployment
deployments "nginx-deployment"
REVISION    CHANGE-CAUSE
1           kubectl create -f nginx-deployment.yaml --record
2           kubectl edit deployment/nginx-deployment
3           kubectl set image deployment/nginx-deployment nginx=nginx:1.91

		- 
$ kubectl rollout history deployment/nginx-deployment --revision=2

	- 回滚Deployment 的Pod的版本

		- $ kubectl rollout undo deployment/nginx-deployment
deployment.extensions/nginx-deployment

		- 
$ kubectl rollout undo deployment/nginx-deployment --to-revision=2
deployment.extensions/nginx-deployment

	- 多次更新操作，只生成一个ReplicaSet

		- 1. kubectl rollout pause deployment/nginx-deployment

			- 让Deployment 进入了一个“暂停”状态，之后对deployment修改不会触发新的“滚动更新”，不会生成新的ReplicaSet

		- 2. kubectl rollout resume deployment/nginx-deployment

			- 恢复Deployment

	- 控制“历史”ReplicaSet 的数量

		- spec.revisionHistoryLimit

			- 如果设置会0，即无法进行回滚操作

### StatefulSet

- 场景

	- 拓扑状态

		- 应用的多个实例之间不是完全对等的关系。这些应用实例，必须按照某些顺序启动，比如应用的主节点 A 要先于从节点 B 启动。而如果你把 A 和 B 两个 Pod 删除掉，它们再次被创建出来时也必须严格按照这个顺序才行。并且，新创建出来的 Pod，必须和原来 Pod 的网络标识一样，这样原先的访问者才能使用同样的方法，访问到这个新 Pod

	- 存储状态

		- 应用的多个实例分别绑定了不同的存储数据。对于这些应用实例来说，Pod A 第一次读取到的数据，和隔了十分钟之后再次读取到的数据，应该是同一份，哪怕在此期间 Pod A 被重新创建过。这种情况最典型的例子，就是一个数据库应用的多个存储实例

- 核心功能

	- 通过某种方式记录这些状态，然后在 Pod 被重新创建时，能够为新 Pod 恢复这些状态

- 工作原理

	- 1. StatefulSet 的控制器直接管理的是 Pod

		- 这是因为，StatefulSet 里的不同 Pod 实例，不再像 ReplicaSet 中那样都是完全一样的，而是有了细微区别的。比如，每个 Pod 的 hostname、名字等都是不同的、携带了编号的。而 StatefulSet 区分这些实例的方式，就是通过在 Pod 的名字里加上事先约定好的编号

	- 2. Kubernetes 通过 Headless Service，为这些有编号的 Pod，在 DNS 服务器中生成带有同样编号的 DNS 记录

		- 只要 StatefulSet 能够保证这些 Pod 名字里的编号不变，那么 Service 里类似于 web-0.nginx.default.svc.cluster.local 这样的 DNS 记录也就不会变，而这条记录解析出来的 Pod 的 IP 地址，则会随着后端 Pod 的删除和再创建而自动更新。这当然是 Service 机制本身的能力，不需要 StatefulSet 操心

	- 3. StatefulSet 还为每一个 Pod 分配并创建一个同样编号的 PVC

		- 这样，Kubernetes 就可以通过 Persistent Volume 机制为这个 PVC 绑定上对应的 PV，从而保证了每一个 Pod 都拥有一个独立的 Volume

- 拓扑状态场景

	- 编写YAML

		- apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: web
spec:
  serviceName: "nginx"
  replicas: 2
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.9.1
        ports:
        - containerPort: 80
          name: web

			- 和nginx-deployment 的唯一区别，就是多了一个 serviceName=nginx 字段

			- serviceName=nginx作用

				- 就是告诉 StatefulSet 控制器，在执行控制循环（Control Loop）的时候，请使用 nginx 这个 Headless Service 来保证 Pod 的“可解析身份”

				- 通过 Headless Service 的方式，StatefulSet 为每个 Pod 创建了一个固定并且稳定的 DNS 记录，来作为它的访问入口

		- 创建

			- 
$ kubectl create -f svc.yaml
$ kubectl get service nginx
NAME      TYPE         CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
nginx     ClusterIP    None         <none>        80/TCP    10s

$ kubectl create -f statefulset.yaml
$ kubectl get statefulset web
NAME      DESIRED   CURRENT   AGE
web       2         1         19s

				- 查看 StatefulSet 创建两个有状态实例的过程

					- 
$ kubectl get pods -w -l app=nginx
NAME      READY     STATUS    RESTARTS   AGE
web-0     0/1       Pending   0          0s
web-0     0/1       Pending   0         0s
web-0     0/1       ContainerCreating   0         0s
web-0     1/1       Running   0         19s
web-1     0/1       Pending   0         0s
web-1     0/1       Pending   0         0s
web-1     0/1       ContainerCreating   0         0s
web-1     1/1       Running   0         20s

						- Pod 的创建，是严格按照编号顺序进行的。比如，在 web-0 进入到 Running 状态、并且细分状态（Conditions）成为 Ready 之前，web-1 会一直处于 Pending 状态

				- 查看pod启动后的hostname

					- $ kubectl exec web-0 -- sh -c 'hostname'
web-0
$ kubectl exec web-1 -- sh -c 'hostname'
web-1

				- 以 DNS 的方式，访问Headless Service

					- 
$ kubectl run -i --tty --image busybox:1.28.4 dns-test --restart=Never --rm /bin/sh
$ nslookup web-0.nginx
Server:    10.0.0.10
Address 1: 10.0.0.10 kube-dns.kube-system.svc.cluster.local

Name:      web-0.nginx
Address 1: 10.244.1.7

$ nslookup web-1.nginx
Server:    10.0.0.10
Address 1: 10.0.0.10 kube-dns.kube-system.svc.cluster.local

Name:      web-1.nginx
Address 1: 10.244.2.7

- 存储状态场景

	- 编写YAML

		- 
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: web
spec:
  serviceName: "nginx"
  replicas: 2
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.9.1
        ports:
        - containerPort: 80
          name: web
        volumeMounts:
        - name: www
          mountPath: /usr/share/nginx/html
  volumeClaimTemplates:
  - metadata:
      name: www
    spec:
      accessModes:
      - ReadWriteOnce
      resources:
        requests:
          storage: 1Gi

			- 作用

				- 凡是被这个 StatefulSet 管理的 Pod，都会声明一个对应的 PVC；而这个 PVC 的定义，就来自于 volumeClaimTemplates 这个模板字段。更重要的是，这个 PVC 的名字，会被分配一个与这个 Pod 完全一致的编号

			- 步骤

				- 子主题 1

- 命令

	- 实时查看 StatefulSet 创建两个有状态实例的过程

		- kubectl get pods -w -l app=nginx

	- 修改StatuefulSet

		- kubectl -n ti-base edit statefulset  (statefulset 名称)

			- kubectl -n ti-base edit statefulset ti-data-center-backend

				- ti-data-center-backend-cm

	- 删除pod

		- $ kubectl delete pod -l app=nginx
pod "web-0" deleted
pod "web-1" deleted

	- 以“补丁”的方式（JSON 格式的）修改一个 API 对象的指定字段

		- $ kubectl patch statefulset mysql --type='json' -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/image", "value":"mysql:5.7.23"}]'
statefulset.apps/mysql patched

	- 指定的一部分实例不会被更新到最新的版本

		- $ kubectl patch statefulset mysql -p '{"spec":{"updateStrategy":{"type":"RollingUpdate","rollingUpdate":{"partition":2}}}}'
statefulset.apps/mysql patched

			- 操作等同于直接使用 kubectl edit 命令，打开这个对象，把 partition 字段修改为 2

			- 只有序号大于或者等于 2 的 Pod 会被更新到这个版本。并且，如果你删除或者重启了序号小于 2 的 Pod，等它再次启动后，也会保持原先的 5.7.2 版本，绝不会被升级到 5.7.23 版本

### DaemonSet

- pod特点

	- 这个 Pod 运行在 Kubernetes 集群里的每一个节点（Node）上

	- 每个节点上只有一个这样的 Pod 实例

	- 当有新的节点加入 Kubernetes 集群后，该 Pod 会自动地在新节点上被创建出来；而当旧节点被删除后，它上面的 Pod 也相应地会被回收掉

- API定义

	- apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd-elasticsearch
  namespace: kube-system
  labels:
    k8s-app: fluentd-logging
spec:
  selector:
    matchLabels:
      name: fluentd-elasticsearch
  template:
    metadata:
      labels:
        name: fluentd-elasticsearch
    spec:
      tolerations:
      - key: node-role.kubernetes.io/master
        effect: NoSchedule
      containers:
      - name: fluentd-elasticsearch
        image: k8s.gcr.io/fluentd-elasticsearch:1.20
        resources:
          limits:
            memory: 200Mi
          requests:
            cpu: 100m
            memory: 200Mi
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
      terminationGracePeriodSeconds: 30
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers

		- DaemonSet 跟 Deployment 其实非常相似，只不过是没有 replicas 字段；也使用 selector 选择管理所有携带了 name=fluentd-elasticsearch 标签的 Pod

		- Pod 的模板，也是用 template 字段定义的。在这个字段中，我们定义了一个使用 fluentd-elasticsearch:1.20 镜像的容器

		- fluentd 启动之后，它会从这两个目录里搜集日志信息，并转发给 ElasticSearch 保存。这样，我们通过 ElasticSearch 就可以很方便地检索这些日志了

		- 需要注意的是，Docker 容器里应用的日志，默认会保存在宿主机的 /var/lib/docker/containers/{{. 容器 ID}}/{{. 容器 ID}}-json.log 文件里，所以这个目录正是 fluentd 的搜集目标

- DaemonSet 如何保证每个 Node 上有且只有一个被管理的 Pod

	- DaemonSet Controller，首先从 Etcd 里获取所有的 Node 列表，然后遍历所有的 Node

	- 检查当前这个 Node 上是不是有一个携带了 name=fluentd-elasticsearch 标签的 Pod 在运行

		- 没有这种 Pod，那么就意味着要在这个 Node 上创建这样一个 Pod

			- 如何在指定的 Node 上创建新 Pod

				- nodeSelector（将要被废弃）

					- nodeSelector:
    name: <Node名字>

				- nodeAffinity

					- apiVersion: v1
kind: Pod
metadata:
  name: with-node-affinity
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: metadata.name
            operator: In
            values:
            - node-geektime

						- 此处定义的 nodeAffinity 的含义

							- requiredDuringSchedulingIgnoredDuringExecution：它的意思是说，这个 nodeAffinity 必须在每次调度的时候予以考虑。同时，这也意味着你可以设置在某些情况下不考虑这个 nodeAffinity

							- 这个 Pod，将来只允许运行在“metadata.name”是“node-geektime”的节点上

		- 有这种 Pod，但是数量大于 1，那就说明要把多余的 Pod 从这个 Node 上删除掉

			- 直接调用 Kubernetes API

		- 正好只有一个这种 Pod，那说明这个节点是正常的

- DaemonSet Controller 会在创建 Pod时，注意点

	- DaemonSet 并不需要修改用户提交的 YAML 文件里的 Pod 模板，而是在向 Kubernetes 发起请求之前，直接修改根据模板生成的 Pod 对象

	- DaemonSet 会给这个 Pod 自动加上另外一个与调度相关的字段，叫作 tolerations。这个字段意味着这个 Pod，会“容忍”（Toleration）某些 Node 的“污点”（Taint）

		- apiVersion: v1
kind: Pod
metadata:
  name: with-toleration
spec:
  tolerations:
  - key: node.kubernetes.io/unschedulable
    operator: Exists
    effect: NoSchedule

			- 此处Toleration 的含义

				- “容忍”所有被标记为 unschedulable“污点”的 Node；“容忍”的效果是允许调度

					- 这种机制，正是我们在部署 Kubernetes 集群的时候，能够先部署 Kubernetes 本身、再部署网络插件的根本原因：因为当时我们所创建的 Weave 的 YAML，实际上就是一个 DaemonSet

				- 在正常情况下，被标记了 unschedulable“污点”的 Node，是不会有任何 Pod 被调度上去的（effect: NoSchedule

- DaemonSet 控制器操作的直接就是 Pod，不可能有 ReplicaSet 这样的对象参与其中。那么，它的这些版本又是如何维护的呢

	- ControllerRevision（专门用来记录某种 Controller 对象的版本）

		- $ kubectl get controllerrevision -n kube-system -l name=fluentd-elasticsearch
NAME                               CONTROLLER                             REVISION   AGE
fluentd-elasticsearch-64dc6799c9   daemonset.apps/fluentd-elasticsearch   2          1h

### Job

- 功能

	- 一次性任务

- 工作原理

	- Job Controller 控制的对象，直接就是 Pod

	- Job Controller 在控制循环中进行的调谐（Reconcile）操作，是根据实际在 Running 状态 Pod 的数目、已经成功退出的 Pod 的数目，以及 parallelism、completions 参数的值共同计算出在这个周期里，应该创建或者删除的 Pod 数目，然后调用 Kubernetes API 来执行这个操作

- api定义

	- apiVersion: batch/v1
kind: Job
metadata:
  name: pi
spec:
  template:
    spec:
      containers:
      - name: pi
        image: resouer/ubuntu-bc 
        command: ["sh", "-c", "echo 'scale=10000; 4*a(1)' | bc -l "]
      restartPolicy: Never
  backoffLimit: 4

		- Job 对象在创建后，它的 Pod 模板，被自动加上了一个 controller-uid=< 一个随机字符串 > 这样的 Label

			- 作用

				- 为了避免不同 Job 对象所管理的 Pod 发生重合

				- 这种自动生成的 Label 对用户来说并不友好，所以不太适合推广到 Deployment 等长作业编排对象上

		- 而这个 Job 对象本身，则被自动加上了这个 Label 对应的 Selector，从而 保证了 Job 与它所管理的 Pod 之间的匹配关系

		- restartPolicy=Never 的原因：离线计算的 Pod 永远都不应该被重启，否则它们会再重新计算一遍

	- 字段

		- spec:
 backoffLimit: 5
 activeDeadlineSeconds: 100

			- 一旦运行超过了 100 s，这个 Job 的所有 Pod 都会被终止

		- 并行控制

			- spec.parallelism

				- 它定义的是一个 Job 在任意时间最多可以启动多少个 Pod 同时运行，即job同时运行的pod数

			- spec.completions

				- 它定义的是 Job 至少要完成的 Pod 数目，即 Job 的最小完成数

			- 需要创建的 Pod 数目 = 最终需要的 Pod 数目（completions） - 实际在 Running 状态 Pod 数目 - 已经成功退出的 Pod 数目

				- 如果需要创建的 Pod 数目 （x）< parallelism ， 那么只会创建x个

- 使用 Job 对象的方法

	- 第一种用法，也是最简单粗暴的用法：外部管理器 +Job 模板

		- yaml

			- 
apiVersion: batch/v1
kind: Job
metadata:
  name: process-item-$ITEM
  labels:
    jobgroup: jobexample
spec:
  template:
    metadata:
      name: jobexample
      labels:
        jobgroup: jobexample
    spec:
      containers:
      - name: c
        image: busybox
        command: ["sh", "-c", "echo Processing item $ITEM && sleep 5"]
      restartPolicy: Never

		- 通过脚本工具来对模版进行修改为所需要的

		- 注意点

			- 创建 Job 时，替换掉 $ITEM 这样的变量

				- 命令

					- 
$ mkdir ./jobs
$ for i in apple banana cherry
do
  cat job-tmpl.yaml | sed "s/\$ITEM/$i/" > ./jobs/job-$i.yaml
done

						- 
$ kubectl create -f ./jobs
$ kubectl get pods -l jobgroup=jobexample
NAME                        READY     STATUS      RESTARTS   AGE
process-item-apple-kixwv    0/1       Completed   0          4m
process-item-banana-wrsf7   0/1       Completed   0          4m
process-item-cherry-dnfu9   0/1       Completed   0          4m

			- 所有来自于同一个模板的 Job，都有一个 jobgroup: jobexample 标签，也就是说这一组 Job 使用这样一个相同的标识

	- 第二种用法：拥有固定任务数目的并行 Job

		- 只关心最后是否有指定数目（spec.completions）个任务成功退出。至于执行时的并行度是多少，我并不关心

		- yaml

			- 
apiVersion: batch/v1
kind: Job
metadata:
  name: job-wq-1
spec:
  completions: 8
  parallelism: 2
  template:
    metadata:
      name: job-wq-1
    spec:
      containers:
      - name: c
        image: myrepo/job-wq-1
        env:
        - name: BROKER_URL
          value: amqp://guest:guest@rabbitmq-service:5672
        - name: QUEUE
          value: job1
      restartPolicy: OnFailure

	- 第三种用法，也是很常用的一个用法：指定并行度（parallelism），但不设置固定的 completions 的值

		- 此时，就必须自己想办法，来决定什么时候启动新 Pod，什么时候 Job 才算执行完成。在这种情况下，任务的总数是未知的，所以你不仅需要一个工作队列来负责任务分发，还需要能够判断工作队列已经为空（即：所有的工作已经结束了）

		- yaml

			- apiVersion: batch/v1
kind: Job
metadata:
  name: job-wq-2
spec:
  parallelism: 2
  template:
    metadata:
      name: job-wq-2
    spec:
      containers:
      - name: c
        image: gcr.io/myproject/job-wq-2
        env:
        - name: BROKER_URL
          value: amqp://guest:guest@rabbitmq-service:5672
        - name: QUEUE
          value: job2
      restartPolicy: OnFailure

		- pod代码决定自己何时结束

			- 
/* job-wq-2的伪代码 */
for !queue.IsEmpty($BROKER_URL, $QUEUE) {
  task := queue.Pop()
  process(task)
}
print("Queue empty, exiting")
exit

- 命令

	- 创建Job

		- kubectl create -f job.yaml

	- 获取jpb信息

		- $ kubectl get job
NAME      DESIRED   SUCCESSFUL   AGE
pi        4         0            3s

	- 查看Job 对象

		- kubectl describe jobs/pi

	- 查看job创建的pod状态

		- $ kubectl get pods
NAME                                READY     STATUS      RESTARTS   AGE
pi-rq5rl                            0/1       Completed   0          4m

	- 查看Pod日志

		- kubectl logs pi-rq5rl

### CronJob

- 功能

	- 定时任务

- api定义

	- apiVersion: batch/v1beta1
kind: CronJob
metadata:
  name: hello
spec:
  schedule: "*/1 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: hello
            image: busybox
            args:
            - /bin/sh
            - -c
            - date; echo Hello from the Kubernetes cluster
          restartPolicy: OnFailure

- 与Job关系

	- CronJob 是一个专门用来管理 Job 对象的控制器

	- CronJob创建和删除 Job 的依据，是 schedule 字段定义的、一个标准的Unix Cron格式的表达式

- 处理策略

	- 现象

		- 由于定时任务的特殊性，很可能某个 Job 还没有执行完，另外一个新 Job 就产生了

	- 控制字段

		- spec.concurrencyPolic

			- concurrencyPolicy=Allow

				- 默认情况，这意味着这些 Job 可以同时存在

			- concurrencyPolicy=Forbid

				- 不会创建新的 Pod，该创建周期被跳过

			- concurrencyPolicy=Replace

				- 新产生的 Job 会替换旧的、没有执行完的 Job

		- spec.startingDeadlineSeconds

			- startingDeadlineSeconds=200

				- 意味着在过去 200 s 里，如果 miss 的数目达到了 100 次，那么这个 Job 就不会被创建执行了

- 命令

	- 创建CronJob

		- kubectl create -f ./cronjob.yaml

	- 获取CronJob信息

		- kubectl get cronjob hello

### 声明式API

- kubectl apply

	- 与kubectl replace 命令区别

		- kubectl replace 的执行过程，是使用新的 YAML 文件中的 API 对象，替换原有的 API 对象

		- 而 kubectl apply，则是执行了一个对原有 API 对象的 PATCH 操作

		- 这意味着 kube-apiserver 在响应命令式请求（比如，kubectl replace）的时候，一次只能处理一个写请求，否则会有产生冲突的可能。而对于声明式请求（比如，kubectl apply），一次能处理多个写操作，并且具备 Merge 能力

- 特点

	- 首先，所谓“声明式”，指的就是我只需要提交一个定义好的 API 对象来“声明”，我所期望的状态是什么样子

	- 其次，“声明式 API”允许有多个 API 写端，以 PATCH 的方式对 API 对象进行修改，而无需关心本地原始 YAML 文件的内容

	- 最后，也是最重要的，有了上述两个能力，Kubernetes 项目才可以基于对 API 对象的增、删、改、查，在完全无需外界干预的情况下，完成对“实际状态”和“期望状态”的调谐（Reconcile）过程

- Kubernetes 编程范式

	- 如何使用控制器模式，同 Kubernetes 里 API 对象的“增、删、改、查”进行协作，进而完成用户业务逻辑的编写过程

## kubernetes Applications Management 应用配置

### Service

- 作用

	- 将一组 Pod 暴露给外界访问的一种机制，用户只要访问这个service，就能访问具体到具体的pod

- 原理

	- service由kube-proxy 和 iptables共同实现

	- 流程

		- 创新一个新的service 后，kube-proxy通过service的infomer感知到新服务被添加

		- kube-proxy感知后，创建一个新的iptable规则，可通过iptable-save命令查看

		- # 凡是目的ip是10.0.1.175，目的端口是80的流量都转向KUBE-SVC-NWV5X2332I4OT4T3 iptable连进行处理
-A KUBE-SERVICES -d 10.0.1.175/32 -p tcp -m comment --comment "default/hostnames: cluster IP" -m tcp --dport 80 -j KUBE-SVC-NWV5X2332I4OT4T3

		- k8s的iptble链其实是DNAT规则·，在路由之前，将目的地的ip和port改为to-destination指定的新目的地和端口，这样就变成访问具体后端pod的地址了
#  具体内部命令
-A KUBE-SEP-57KPRZ3JQVENLNBR -s 10.244.3.6/32 -m comment --comment "default/hostnames:" -j MARK --set-xmark 0x00004000/0x00004000
-A KUBE-SEP-57KPRZ3JQVENLNBR -p tcp -m comment --comment "default/hostnames:" -m tcp -j DNAT --to-destination 10.244.3.6:9376

-A KUBE-SEP-WNBA2IHDGP2BOBGZ -s 10.244.1.7/32 -m comment --comment "default/hostnames:" -j MARK --set-xmark 0x00004000/0x00004000
-A KUBE-SEP-WNBA2IHDGP2BOBGZ -p tcp -m comment --comment "default/hostnames:" -m tcp -j DNAT --to-destination 10.244.1.7:9376

-A KUBE-SEP-X3P2623AGDH6CDF3 -s 10.244.2.3/32 -m comment --comment "default/hostnames:" -j MARK --set-xmark 0x00004000/0x00004000
-A KUBE-SEP-X3P2623AGDH6CDF3 -p tcp -m comment --comment "default/hostnames:" -m tcp -j DNAT --to-destination 10.244.2.3:9376

- 模式

	- ClusterIP 模式

		- 访问方式

			- 以 Service 的 VIP（Virtual IP，即：虚拟 IP）方式

				- 当访问 10.0.23.1 这个 Service 的 IP 地址时，10.0.23.1 其实就是一个 VIP，它会把请求转发到该 Service 所代理的某一个 Pod 上

			- 以 Service 的 DNS 方式

				- 作用

					- 访问“my-svc.my-namespace.svc.cluster.local”这条 DNS 记录，就可以访问到名叫 my-svc 的 Service 所代理的某一个 Pod

				- 2种实现方式

					- Normal Service

						- 访问“my-svc.my-namespace.svc.cluster.local”解析到的，正是 my-svc 这个 Service 的 VIP，后面的流程就跟 VIP 方式一致了

					- Headless Service

						- 访问“my-svc.my-namespace.svc.cluster.local”解析到的，直接就是 my-svc 代理的某一个 Pod 的 IP 地址

							- apiVersion: v1
kind: Service
metadata:
  name: nginx
  labels:
    app: nginx
spec:
  ports:
  - port: 80
    name: web
  clusterIP: None
  selector:
    app: nginx

							- <pod-name>.<svc-name>.<namespace>.svc.cluster.local

						- 可以看到，这里的区别在于，Headless Service 不需要分配一个 VIP，而是可以直接以 DNS 记录的方式解析出被代理 Pod 的 IP 地址

	- 子主题 2

### Ingress

- 4层服务发现（TCP和UDP协议转发）

	- 基于 IP+端口的负载均衡

	- 在三层负载均衡的基础上，通过发布三层的 IP 地 址(VIP)，然后加四层的端口号，来决定哪些流量需要做负载均衡，对需要处理的流量进行 NAT 处理， 转发至后台服务器，并记录下这个 TCP 或者 UDP 的流量是由哪台服务器处理的，后续这个连接的所有流 量都同样转发到同一台服务器处理

- 7层服务发现（HTTP和HTTPS协议转发）

	- 基于URL 或主机 IP 的负载均衡

	- 在四层负载均衡的基础上(没有四层是绝对不可能有七层的)，再考虑应用层的特征，比如同一个 Web 服务器的负载均衡，除了根据 VIP 加 80 端口辨别是否需要处理的流量，还可根据七层的 URL、浏览器类别、语言来决定是否要进行负载均衡。举个例子，如果你的 Web 服务器分成两组，一组是中文语言的，一组是英文语言的，那么七层负载 均衡就可以当用户来访问你的域名时，自动辨别用户语言，然后选择对应的语言服务器组进行负载均衡处理

	- 子主题 3

	- 子主题 4

	- 子主题 5

### Projected Volume

- Secret

	- 一般保存需要加密的，应用所需的配置

		- 
apiVersion: v1
kind: Pod
metadata:
  name: test-projected-volume 
spec:
  containers:
  - name: test-secret-volume
    image: busybox
    args:
    - sleep
    - "86400"
    volumeMounts:
    - name: mysql-cred
      mountPath: "/projected-volume"
      readOnly: true
  volumes:
  - name: mysql-cred
    projected:
      sources:
      - secret:
          name: user
      - secret:
          name: pass

	- 问题

		- secret是Base64的编码方式（不是加密方式），为什么说Secret是安全的呢

			- 传输安全（K8S中与API Server的交互都是HTTPS的）

			- 存储安全（Secret被挂载到容器时存储在tmpfs中，只存在于内存中而不是磁盘中，Pod销毁Secret随之消失）

			- 访问安全（Pod间的Secret是隔离的，一个Pod不能访问另一个Pod的Secret）

- ConfigMap

	- 一般保存的是不需要加密的，应用所需的配置

		- 
# .properties文件的内容
$ cat example/ui.properties
color.good=purple
color.bad=yellow
allow.textmode=true
how.nice.to.look=fairlyNice

# 从.properties文件创建ConfigMap
$ kubectl create configmap ui-config --from-file=example/ui.properties

# 查看这个ConfigMap里保存的信息(data)
$ kubectl get configmaps ui-config -o yaml
apiVersion: v1
data:
  ui.properties: |
    color.good=purple
    color.bad=yellow
    allow.textmode=true
    how.nice.to.look=fairlyNice
kind: ConfigMap
metadata:
  name: ui-config
  ...

	- 挂载subPath

		- 界面操作

			- 数据卷

				-  

					- cm.png

				-  

					- cm2.png

					- 把conf中的net_param.cfg拷贝到/app/third_path/xxx/net_param.cfg

		- yaml文件

			-  

				- yaml-1.png

			-  

				- yaml-2.png

- Downward API

	- 让 Pod 里的容器能够直接获取到这个 Pod API 对象本身的信息

- ServiceAccountToken

	- Service Account 对象的作用，就是 Kubernetes 系统内置的一种“服务账户”，它是 Kubernetes 进行权限分配的对象。比如，Service Account A，可以只被允许对 Kubernetes API 进行 GET 操作，而 Service Account B，则可以有 Kubernetes API 的所有操作权限

	- 任何运行在 Kubernetes 集群上的应用，都必须使用这个 ServiceAccountToken 里保存的授权信息，也就是 Token，才可以合法地访问 API Server

	- 每一个 Pod，都已经自动声明一个类型是 Secret、名为 default-token-xxxx 的 Volume，然后 自动挂载在每个容器的一个固定目录上

## 扩展和插件

### Istio

- 功能

	- 微服务路由和负载均衡

- 架构

	-  

		- 3.jpg

- 原理

	- 把Envoy把这个代理服务以 sidecar 容器的方式，运行在了每一个被治理的应用 Pod 中。因为Pod 里的所有容器都共享同一个 Network Namespace。所以，Envoy 容器就能够通过配置 Pod 里的 iptables 规则，把整个 Pod 的进出流量接管下来。

	- Istio 的控制层（Control Plane）里的 Pilot 组件，就能够通过调用每个 Envoy 容器的 API，对这个 Envoy 代理进行配置，从而实现微服务治理

- 核心

	- 由无数个运行在应用 Pod 中的 Envoy 容器组成的服务代理网格。这也正是 Service Mesh 的含义

- 问题

	- Envoy能够击败 Nginx 以及 HAProxy 等竞品，成为 Service Mesh 体系的核心原因？

		- envoy提供了api形式的配置入口，更方便做流量治理

	- Istio 项目明明需要在每个 Pod 里安装一个 Envoy 容器，如何做到“无感”的呢？

		- Dynamic Admission Control，“热插拔”式的 Admission 机制（也叫作：Initializer）

			- 在pod yaml里自动加上Envoy 容器的配置

				- 类似于（加上- name: envoy部分）

					- 
apiVersion: v1
kind: Pod
metadata:
  name: myapp-pod
  labels:
    app: myapp
spec:
  containers:
  - name: myapp-container
    image: busybox
    command: ['sh', '-c', 'echo Hello Kubernetes! && sleep 3600']
  - name: envoy
    image: lyft/envoy:845747b88f102c0fd262ab234308e9e22f693a1
    command: ["/usr/local/bin/envoy"]
    ...

				- 原理

					- 把Envoy 相关的字段，自动添加到用户提交的 Pod 的 API 对象里，在 Initializer 更新用户的 Pod 对象的时候，必须使用 PATCH API 来完成

					- 接下来，Istio 将一个编写好的 Initializer，作为一个 Pod 部署在 Kubernetes 中。

						- apiVersion: v1
kind: Pod
metadata:
  labels:
    app: envoy-initializer
  name: envoy-initializer
spec:
  containers:
    - name: envoy-initializer
      image: envoy-initializer:0.0.1
      imagePullPolicy: Always

						- envoy-initializer控制器的作用

							- 实际上就是一个“死循环”：它不断地获取“实际状态”，然后与“期望状态”作对比，并以此为依据决定下一步的操作

							- 不断获取到的“实际状态”，就是用户新创建的 Pod。而它的“期望状态”，则是：这个 Pod 里被添加了 Envoy 容器的定义

							- 
for {
  // 获取新创建的Pod
  pod := client.GetLatestPod()
  // Diff一下，检查是否已经初始化过
  if !isInitialized(pod) {
    // 没有？那就来初始化一下
    doSomething(pod)
  }
}

								- Initializer 控制器的工作逻辑

									- 首先会从 APIServer 中拿到envoy-initializer的 ConfigMap

										- 
func doSomething(pod) {
  cm := client.Get(ConfigMap, "envoy-initializer")
}

									- 然后，把这个 ConfigMap 里存储的 containers 和 volumes 字段，直接添加进一个空的 Pod 对象里

										- 
func doSomething(pod) {
  cm := client.Get(ConfigMap, "envoy-initializer")
  
  newPod := Pod{}
  newPod.Spec.Containers = cm.Containers
  newPod.Spec.Volumes = cm.Volumes
}

									- 直接使用新旧两个 Pod 对象，生成一个 TwoWayMergePatch，这样，一个用户提交的 Pod 对象里，就会被自动加上 Envoy 容器相关的字段

										- 
func doSomething(pod) {
  cm := client.Get(ConfigMap, "envoy-initializer")

  newPod := Pod{}
  newPod.Spec.Containers = cm.Containers
  newPod.Spec.Volumes = cm.Volumes

  // 生成patch数据
  patchBytes := strategicpatch.CreateTwoWayMergePatch(pod, newPod)

  // 发起PATCH请求，修改这个pod对象
  client.Patch(pod.Name, patchBytes)
}

### Helm

## 持久化存储

### pvc/pv

- 定义

	- pv

		- 一个具体的volume的属性，比如volume类型、挂载目录、远程服务器的地址等

			- apiVersion: v1
kind: PersistentVolume
metadata:
  name: nfs
spec:
  storageClassName: manual
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteMany
  nfs:
    server: 10.244.1.4
    path: "/"

	- pvc

		- pod想要持久化存储的属性，比如存储的大小、读写权限等

			- apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nfs
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: manual
  resources:
    requests:
      storage: 1Gi

- 使用pvc/pv步骤

	- 使用pvc步骤

		- 定义一个 PVC，声明想要的 Volume 的属性

			- kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: pv-claim
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi

		- 在应用的 Pod 中，声明使用这个 PVC

			- 
apiVersion: v1
kind: Pod
metadata:
  name: pv-pod
spec:
  containers:
    - name: pv-container
      image: nginx
      ports:
        - containerPort: 80
          name: "http-server"
      volumeMounts:
        - mountPath: "/usr/share/nginx/html"
          name: pv-storage
  volumes:
    - name: pv-storage
      persistentVolumeClaim:
        claimName: pv-claim

				- 创建这个 PVC 对象，Kubernetes 就会自动为它绑定一个符合条件的 Volume

		- PV/PVC绑定

			- 条件

				- PV和PVC的spec字段，比如PV的存储大小必须满足PVC的要求

				- PV和PVC的storageClassName字段必须一样

			- 过程

				- PersistentVolumeController

					- 不断查看当前每个PVC是否处于Bound(绑定)状态

					- 如果没有绑定，就会遍历所有、可用的PV，尝试进行绑定

				- 效果

					- 将PV的对象名字，填在PVC对象的spec.volumeName字段上

	- PV编写

		- kind: PersistentVolume
apiVersion: v1
metadata:
  name: pv-volume
  labels:
    type: local
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  rbd:
    monitors:
    # 使用 kubectl get pods -n rook-ceph 查看 rook-ceph-mon- 开头的 POD IP 即可得下面的列表
    - '10.16.154.78:6789'
    - '10.16.154.82:6789'
    - '10.16.154.83:6789'
    pool: kube
    image: foo
    fsType: ext4
    readOnly: true
    user: admin
    keyring: /etc/ceph/keyring

	- PVC/PV联系

		- PVC 和 PV 的设计，实际上类似于“接口”和“实现”的思想。开发者只要知道并会使用“接口”，即：PVC；而运维人员则负责给“接口”绑定具体的实现，即：PV

	- 命令

		- 查看pvc

			- kubectl get pvc -l app=nginx

				- PVC命名规则：< 编号 >”的方式命名，并且处于 Bound 状态

		- 在多个pod里写文件

			- $ for i in 0 1; do kubectl exec web-$i -- sh -c 'echo hello $(hostname) > /usr/share/nginx/html/index.html'; done

## 命令

### docker

- 登录

	- # 地址不填将登陆默认地址
docker login --username=admin docker.com

- 镜像操作

	- 从registry拉取镜像

		- docker pull docker.com/stream:[tag]

	- 将镜像推送到registry

		- docker push docker.com/stream:[tag]

	- 打tag

		- docker tag [ImageId] docker.com/stream:[tag]

	- 镜像导入/导出

		- 导出

			- docker save -o nginx.docker.arm64.tar.gz arm64v8/nginx:latest

		- 导入

			- docker load <  nginx.docker.arm64.tar.gz

	- 容器保存为镜像

		- docker commit 容器Id 仓库名:标签
docker commit d47d924d6410  wxg-dev-last:latest

	- 删除

		- # 移除没有标签并且没有被容器引用的镜像
docker image prune

		- # 移除所有没有容器使用的镜像
docker image prune -a

- 拷贝

	- 宿主机->容器

		- docker cp /home/runoob 96f7f14e99ab:/home/

	- 容器->宿主机

		- docker cp  96f7f14e99ab:/www /tmp/

- 查看容器历史启动命令

	- [$ sudo pip install runlike

# run the ubuntu image
$ docker run -ti ubuntu bash

$ docker ps -a  
# suppose you get the container ID 1dfff2ba0226

# Run runlike to get the docker run command. 
$ runlike 1dfff2ba0226
docker run --name=elated_cray -t ubuntu bash](https://stackoverflow.com/questions/32758793/how-to-show-the-run-command-of-a-docker-container)

- 查看镜像各个layer信息

	-  docker history --no-trunc  mysql:5.7

- nvidia服务启动

	- docker run -d  --ipc=host -p 8003:8003   --runtime=nvidia --net=host -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video --privileged -v /data/home/license/license.lic:/data/home/license/license.lic -v /data/home/testdata/video:/app/video --name face_cuda10_docker -v /data/conf/face_cuda10_docker:/app/conf -v /data/logs/face_cuda10_docker:/app/logs  face_track:last

### service

- 通过service name获取cluster-ip(正常是vip)

	- # 获取service name
kubectl -n ns-1  get service

	- # 获取cluster-ip
kubectl -n ns-1  get svc  hostname

- 删除service

	- kubectl -n ns-1 delete svc service-1 

### replicate set

- 删除replicate set后，pod将全部删除，且不会恢复

	- kubectl -n ns-1 delete rs rs-1

### pod

- 查看一下这个 Pod 的 API 对象

	- kubectl -n rook get pod website -o yaml

	- # 纯文本格式输出（包含pod的node节点）
kubectl -n rook get pod website -o wide 

	- 获取所有命名空间的pods

		- kubectl get pods  --all-namespaces 

- 删除pod

	- 普通删除

		- kubectl delete pod -l app=nginx

	- 强制删除

		- kubectl -n ti-base delete pod/ti-data-center-backend-0  --force --grace-period=0

### deployment

- 查看升级版本

	- kubectl rollout status deploy web -n  kube-system

- 查看历史版本

	- kubectl rollout history deployment web -n kube-system

- 回滚到上一个版本

	- kubectl rollout undo deploy web -n  kube-system

- 回滚到指定版本

	- kubectl rollout undo deploy web --to-revision=2

### DaemonSet

- 创建DaemonSet 对象

	- $ kubectl create -f fluentd-elasticsearch.yaml

- 查看Kubernetes 集群里的 DaemonSet 对象

	- $ kubectl get ds -n kube-system fluentd-elasticsearch
NAME                    DESIRED   CURRENT   READY     UP-TO-DATE   AVAILABLE   NODE SELECTOR   AGE
fluentd-elasticsearch   2         2         2         2            2           <none>          1h

- 查看每次 Deployment 变更对应的版本

	- $ kubectl rollout history daemonset fluentd-elasticsearch -n kube-system
daemonsets "fluentd-elasticsearch"
REVISION  CHANGE-CAUSE
1         <none>

- 更新DaemonSet 的容器镜像版本到 v2.2.0

	- $ kubectl set image ds/fluentd-elasticsearch fluentd-elasticsearch=k8s.gcr.io/fluentd-elasticsearch:v2.2.0 --record -n=kube-system

		- 第一个 fluentd-elasticsearch 是 DaemonSet 的名字，第二个 fluentd-elasticsearch 是容器的名字

- 查看“滚动更新”的过程

	- $ kubectl rollout status ds/fluentd-elasticsearch -n kube-system

Waiting for daemon set "fluentd-elasticsearch" rollout to finish: 0 out of 2 new pods have been updated...
Waiting for daemon set "fluentd-elasticsearch" rollout to finish: 0 out of 2 new pods have been updated...
Waiting for daemon set "fluentd-elasticsearch" rollout to finish: 1 of 2 updated pods are available...
daemon set "fluentd-elasticsearch" successfully rolled out

- 轮替重启 "frontend" Deployment

	- kubectl rollout restart deployment/frontend

- kubectl describe 查看这个 ControllerRevision 对象

	- kubectl describe controllerrevision fluentd-elasticsearch-64dc6799c9 -n kube-system

- 将DaemonSet 回滚到 Revision=1 时的状态

	- $ kubectl rollout undo daemonset fluentd-elasticsearch --to-revision=1 -n kube-system
daemonset.extensions/fluentd-elasticsearch rolled back

		- 这个 kubectl rollout undo 操作，实际上相当于读取到了 Revision=1 的 ControllerRevision 对象保存的 Data 字段。而这个 Data 字段里保存的信息，就是 Revision=1 时这个 DaemonSet 的完整 API 对象。

		- 在执行完这次回滚完成后，你会发现，DaemonSet 的 Revision 并不会从 Revision=2 退回到 1，而是会增加成 Revision=3

			- 这是因为，一个新的 ControllerRevision 被创建了出来

### job

### CronJob

### pv/pvc

### linqueyun

- mysql登录

	- kubectl get svc -n ti-base|grep proxysql

	- mysql -h ip -u user -p -P 6033

- helm启动

	- http_scheme=http helmfile sync

### debug

- 启动参数设置

	- sleep

		- #运行命令
["/bin/bash","-c","--"]

		- # 运行参数
["while true; do sleep 30; done;"]

	- 服务启动

		- ["cd /usr/local/services/milvus; ./admin/restart.sh all && exec /usr/sbin/init"]

