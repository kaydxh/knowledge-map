# Kafka

## 架构

###  

- [ ](https://kafka.apache.org/32/documentation/streams/architecture)

	- kafka.jpg

## 三高设计

### 高性能

- 网络io

	- 利用linux的零拷贝技术

		- 定义

			- 直接将内核数据发送到网卡存储区，此过程不需要将内核缓冲区数据拷贝用户缓存区

		- 技术实现

			- mmap + write

				- 使用mmap将内核缓冲区与用户空间的缓冲区进行映射，实现内核缓冲区与应用程序内存的共享

			- sendfile

				- 数据可以在内核空间内部进行I/O传输，省去了数据在内核空间和用户空间的来回拷贝

	- 数据压缩

- 磁盘数据寻址

	- 磁盘线性操作

### 高并发

- partition设计，可以将不同的partition分布在不同的物理机上

### 高可用

- partition支持副本，如果一个主partition挂了，可以快速切换副本

## 数据安全性

### 消息模式

- # 消息不会丢失，但可能会重复
At least one

	- 先发消息，再写日志，如果日志写失败，将会触发重新发送消息

- # 消息可能会丢，但不会重复传输
At most once

	- 先写日志，再发消息，如果消息发送失败，就会丢消息

- # 每条消息有且仅会传输一次 
Exactly once

### ISR机制

- 术语

	- AR

		- Assigned Replicas
副本全集（包含leader）

	- OSR

		- out-sync Replicas
和leader消息数相差太大的副本列表

			- # 超过10s没有同步数据(follower主动拉取leader消息)
replica.lag.time.max.ms = 10000

			- # 主副节点相差超过4000条数据
replica.lag.max.message = 4000

	- ISR

		- in-sync-Replicas
和leader消息数相差不大的副本列表（包含leader）

- 作用

	- 主节点挂掉后，快速找到继承者

- 脏节点选举

	- 当follower不和leader消息同步，且被选择为leader

### producer

- producer配置中的request.required.acks

	- # acks=0
producer在isr中的leader已成功收到消息，并得到确认后，再发送下一条消息

	- # acks=1
producer无需等待broker的确认，继续发送下一条消息

	- # acks=2
producer需等待isr中所有followers成功收到消息，且确认后，才发送下一条消息

### broker数据存储机制

- 数据删除策略

	- # 基于时间
log.retention.hours=168

	- # 基于大小
log.retention.bytes=1073741824

### consumer

- # at least once（消息重复情况：处理消息和commit之间consumer挂了）
读完消息，先处理消息，再commit （auto.commit.enable=false）

	- 注意：当手动commit时，实际是对这个consumer进程所占有的所有partition都进行了提交操作，可能导致其他线程正在处理其他partition时，数据丢失（比如t2线程在处理完partition2消息后，consumer crash了）

	- 解决方法

		- 当手动commit时，将所有fetch到的消息放入一个队列里，当队列里的消息全部消费完成后，再统一commit

- # at most once （自动提交消息，可能导致数据丢失，commit和处理消息之间，consumer挂了）
读完消息，先commit，再处理消息 （auto.commit.enable=true）

- # exactly once
引入两阶段提交

## 数据顺序性

### 1个topic对1个partition

- 1个Topic（主题）只创建1个Partition(分区)，对于同一个分区的消息，是有序的

### 1个topic对多个partion

- producer

	- 将业务的key的hash值域partition的个数进行取余，得到partition的值发送

- consumer

	- 每个消费者分配一个消费者组（consumer group)，在同一个消费者组中，每个分区只能被一个消费者消费。这样就可以保证同一个分区内的消息只会被一个消费者消费，消费者之间不会出现竞争消费同一个分区的情况

