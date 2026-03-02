# mysql

## 架构

### 架构图

-  

	- [ ](https://time.geekbang.org/column/article/68319?code=XRKGcnAr%2F7QETBqBoRiCVHkd%2FgwYS5kFpGlO3O5AuhY%3D)

### server层

- 连接器

	- 验证输入的密码是否正确

	- 如果密码验证正确，连接器会在权限表里查询出该连接的权限，之后这个连接里面的权限判断都是依赖此时读到的权限数据

- 查询缓存

	- 从查询缓存里去查询(key为sql语句，value为查询结果)，如果有这个value，就会返回给客户端，如果没有会继续执行

- 分析器

	- 分析器会进行语法分析，得到sql语句的含义，如果此时查询的列字段不存在，也会在该阶段检测出来

- 优化器

	- 优化器会决定具体怎么执行(根据扫描行数、是否使用临时表、是否排序等因素进行综合判断)，如使用哪个索引、join语句决定使用各个表的连接顺序（当前实际情况，优化器作出的选择不一定是最优的）

		- 扫描行数的判断

			- MySQL 在真正开始执行语句之前，并不能精确地知道满足这个条件的记录有多少条，而只能根据统计信息来估算记录数（这个统计信息就是索引的“区分度）

			- 一个索引上不同的值（基数）越多，这个索引的区分度就越好

				- # 查看基数cardinality 列
show index from t

		- 优化器为啥有的时候选择扫描行数多的方案执行

			- 可能原因

				- 扫描行数少的方案，可能需要进行回表，优化器会把这个代价也算进去，尽管这个选择有时不是最优的

				-  select * from t where (a between 1 and 1000)  and (b between 50000 and 100000) order by b limit 1;

					- 优化器选择使用索引 b，是因为它认为使用索引 b 可以避免排序（b 本身是索引，已经是有序的了，如果选择索引 b 的话，不需要再做排序，只需要遍历），所以即使扫描行数多，也判定为代价更小

			- 解决方案

				- 强制使用force index矫正，不过不够优美，且移植性不好

				- 可以考虑修改语句，引导 MySQL 使用我们期望的索引

					- # 现在 order by b,a 这种写法，要求按照 b,a 排序，就意味着使用这两个索引都需要排序。因此，扫描行数成了影响决策的主要条件，于是此时优化器选了只需要扫描 1000 行的索引 a
# order by b limit 1” 改成 “order by b,a limit 1

				- 前面2个方法都不通用，在有些场景下，我们可以新建一个更合适的索引，来提供给优化器做选择，或删掉误用的索引

- 执行器

	- 执行器首先判断该连接对表有没有相关的权限，如果没有会反回没有权限的错误，如果有权限，就会根据表的引擎，去使用对应的引擎接口

- 所有的内置函数

### 引擎层

- 数据的存储与提取

### 注意点：

- 对于一个已经连接成功的连接，权限是固定的，此时再次更改该用户的权限，也无法阻止该连接的权限，需要重新连接，才会使用新的权限

- 客户端连接后，自动断开的耗时默认是8h

- 如果使用了长连接进行查询大量的数据，会导致mysql使用的内存增加，长时间累积可能导致oom，所以最好在每次较大的操作后，通过mysql_reset_connection（5.7版本支持）来重置初始化资源，这个过程不需要重连和重新做权限验证，但是会将连接恢复到刚刚创建完的状态

- 除了静态表（长时间才会更新一次）外，一般不使用查询缓存，因为查询缓存失效的频率很高（只要这个表有个更新，这个表上的所有查询缓存就会失效）

- 慢查询日志里的rows_examined，表示的是调用引擎接口获取行的数量累加的，而每调用一次引擎接口，可能引擎接口内部需要扫描多行，所以引擎扫描行数跟rows_examined 并不是完全相同的

- 慢查询的相关变量slow_query_logs（OFF/ON）、slow_query_log_file（log path）、long_query_time（慢查询的时间阈值，默认10s）

## [存储引擎 ](https://dev.mysql.com/doc/refman/8.0/en/storage-engines.html)

### MyISAM

- MyISAM特点

	- 不支持事物

	- 表锁，不支持行锁

	- 不支持外键

	- 支持fulltext索引

	- 对不会修改的表支持压缩表，减小磁盘占用

	- 不支持崩溃后安全恢复，没有redo log

- 使用场景

	- 适合只读或者读多的场景

### InnoDB

- InnoDB特点

	- 支持事物

	- 支持外键

	- 默认支持行锁

	- 5.6版本后也支持fulltext索引

	- 支持崩溃后的安全恢复，crash-safe，有redo log

- 使用场景

	- 通用场景，适用于需要事物支持的大型数据库

## 锁

### 全局锁

- 对整个数据库实例加锁

	- # 加全局读锁的方法, 让整个库处于只读状态的时候,之后其他线程的以下(数据的增删改、建表、修改表结构以及更新类事务)语句会被阻塞
mysql root@localhost:RUNNOB> Flush tables with read lock (FTWRL)

- 使用场景

	- 做全库逻辑备份。也就是把整库每个表都 select 出来存成文本

- 缺点

	- 如果你在主库上备份，那么在备份期间都不能执行更新，业务基本上就得停摆

	- 如果你在从库上备份，那么备份期间从库不能执行主库同步过来的 binlog，会导致主从延迟

- 问题

	- 使用mysqldump进行备份，使用参数–single-transaction 的时候，导数据之前就会启动一个事务，来确保拿到一致性视图。而由于 MVCC 的支持，这个过程中数据是可以正常更新的。有了这个功能，为什么还需要 FTWRL 呢

		- 原因

			- 一致性读是好，但前提是引擎要支持这个隔离级别， 对于 MyISAM 这种不支持事务的引擎，如果备份过程中有更新，总是只能取到最新的数据，那么就破坏了备份的一致性。这时，我们就需要使用 FTWRL 命令了

	- 既然要全库只读，为什么不使用 set global readonly=true 的方式呢

		- 在有些系统中，readonly 的值会被用来做其他逻辑，比如用来判断一个库是主库还是备库。因此，修改 global 变量的方式影响面更大，不建议使用

		- 在异常处理机制上有差异。如果执行 FTWRL 命令之后由于客户端发生异常断开，那么 MySQL 会自动释放这个全局锁，整个库回到可以正常更新的状态。而将整个库设置为 readonly 之后，如果客户端发生异常，则数据库就会一直保持 readonly 状态，这样会导致整个库长时间处于不可写状态，风险较高

### 表级锁

- 表锁特点

	- 每次操作锁住整张表

		- 共享读锁

		- 独占写锁（排他锁）

	- 无死锁

	- 锁粒度大，发生冲突大概率大，并发度低

- 种类

	- 表锁

		- # 与 FTWRL 类似，可以用 unlock tables 主动释放锁，也可以在客户端断开的时候自动释放
lock tables … read/write

#在线程 A 中执行下面语句，则
# 则其他线程写 t1、读写 t2 的语句都会被阻塞
# 同时，线程 A 在执行 unlock tables 之前，也只能执行读 t1、读写 t2 的操作
lock tables t1 read, t2 write

	- 元数据锁（meta data lock，MDL)

		- 特点

			- MDL 不需要显式使用，在访问一个表的时候会被自动加上

			- MDL 的作用是，保证读写的正确性（你可以想象一下，如果一个查询正在遍历一个表中的数据，而执行期间另一个线程对这个表结构做变更，删了一列，那么查询线程拿到的结果跟表结构对不上，肯定是不行的）

		- 使用规则

			- 当对一个表做增删改查操作的时候，加 MDL 读锁；当要对表做结构变更操作的时候，加 MDL 写锁

			- 读锁之间不互斥，因此你可以有多个线程同时对一张表增删改查

			- 读写锁之间、写锁之间是互斥的，用来保证变更表结构操作的安全性。因此，如果有两个线程要同时给一个表加字段，其中一个要等另一个执行完才能开始执行

			- 注意：事务中的 MDL 锁，在语句执行开始时申请，但是语句结束后并不会马上释放，而会等到整个事务提交后再释放

- 问题

	- 如何安全地给小表加字段

		- 1. 解决长事务，事务不提交，就会一直占着 MDL 锁 在 MySQL 的 information_schema 库的 innodb_trx 表中，可以查到当前执行中的事务。

			- 如果要做 DDL （Data Definition Language，CREATE TABLE/VIEW/INDEX/SYN/CLUSTER语句 ）变更的表刚好有长事务在执行，要考虑先暂停 DDL，或者 kill 掉这个长事务

		- 2. 对于变更的表是一个热点表，虽然数据量不大，但是上面的请求很频繁，而不得不加个字段，此时kill未必管用，因为新的请求马上就来了

			- 比较理想的机制是，在 alter table 语句里面设定等待时间，如果在这个指定的等待时间里面能够拿到 MDL 写锁最好，拿不到也不要阻塞后面的业务语句，先放弃。之后开发人员或者 DBA 再通过重试命令重复这个过程

				- ALTER TABLE tbl_name NOWAIT add column ...
ALTER TABLE tbl_name WAIT N add column ... 

# todo
alter TABLE tcount_tbl  add COLUMN  test_id int not null

### 行锁

- 行锁特点

	- 每次操作锁住一行记录

		- 共享读锁

		- 排他锁

	- 会发生死锁

	- 锁粒度低，发生冲突大概率小，并发度高

	- 两阶段锁协议

		- 在 InnoDB 事务中，行锁是在需要的时候才加上的，但并不是不需要了就立刻释放，而是要等到事务结束时才释放

- 行锁的种类

	- Record Lock

		- 单个记录上的锁

		- 会去锁住索引记录，如果InnoDB存储引擎在建立的时候没有设置任何一个索引，就会使用隐式的主键来进行锁定

	- Gap Lock

		- 间隙锁，锁定一个范围，左开右开区间

	- Next-Key-Lock

		- Gap Lock + Record Lock

			- 锁定一个范围，左开右闭区间

			- 解决幻读问题

				- 幻读：在同一个事物下，连续执行2次相同的sql语句可能导致不同的结果，第二次的sql语句可能会返回之前不存在的行

			- 有了Next-Key-Lock之后，第一次sql语句查询会锁住一个范围，这样其他事务就不能进行增删改操作，第二个sql语句就会得到相同的结果，等到本事务结束后，其他事务才能获取锁资源，继续操作

- 更新逻辑

	- 更新数据的时候，都是先读后写的，而这个读，只能读当前的值，称为“当前读”（current read）

	- 而可重复性读的隔离级别，是对于查询而言的，对于更新数据，都是读当前的最新值

	- select 语句如果加锁，也是当前读

		- mysql> select k from t where id=1 lock in share mode;
mysql> select k from t where id=1 for update;

	- 如果有个事物在执行更新语句，没有立马commit提交（写锁没有释放），那么其他事物的update语句将会block（需要读锁-行锁，而写锁没有释放，因此需要等待）

- 死锁

	- 定义

		- 当并发系统中不同线程出现循环资源依赖，涉及的线程都在等待别的线程释放资源时，就会导致这几个线程都进入无限等待的状态

	- 解决死锁的策略

		- 一种策略是，直接进入等待，直到超时。这个超时时间可以通过参数 innodb_lock_wait_timeout 来设置，默认的innodb_lock_wait_timeout为50s太长，对于在线服务无法接受，如果设置太短，就会导致误伤（本来不是死锁，而是简单的锁等待就会误认为死锁）

		- 另一种策略是，发起死锁检测，发现死锁后，主动回滚死锁链条中的某一个事务，让其他事务得以继续执行。将参数 innodb_deadlock_detect 设置为 on(默认就是on)，表示开启这个逻辑

	- 检测死锁的过程

		- 每当一个事务被锁的时候，就要看看它所依赖的线程有没有被别人锁住，如此循环，最后判断是否出现了循环等待，也就是死锁O(n^2)

			- 对于热点行更新，性能低

				- 死锁检测过程是很耗cpu的，对于热点行更新，就会出现CPU 利用率很高，但是每秒却执行不了几个事务

			- 怎么解决由热点行更新导致的性能问题呢

				- 修改mysql源码，对于相同行的更新，在进入引擎之前排队。这样在 InnoDB 内部就不会有大量的死锁检测工作了

				- 通过将一行改成逻辑上的多行来减少锁冲突，比如原来的一行对应逻辑10行，这样每次冲突概率变成原来的 1/10，可以减少锁等待个数，也就减少了死锁检测的 CPU 消耗

- 问题

	- 如果要删除一个表里面的前 10000 行数据，如何做更好

		- 在一个连接中循环执行 20 次 delete from T limit 500（尽量避免事物太长，锁时间太长）

## 索引

### 哈希表

- 哈希表特点

	- 哈希表适用于等值查询

	- 不支持范围查找

		- 由于 Hash 索引比较的是进行 Hash 运算之后的 Hash 值，所以它只能用于等值的过滤，不能用于基于范围的过滤，因为经过相应的 Hash 算法处理之后的 Hash 值的大小关系，并不能保证和Hash运算前完全一样

		- 在同一个key里范围查找也是效率很低的：因为不可避免的，会存在key冲突，比较常用的方法是使用链表法解决冲突，而链表里面存的是无序的，此时范围查找，就需要全部遍历，效率低

- 使用例子

	- Memcached

	- 其他一些 NoSQL 引擎

### 有序数组

- 有序数组特点

	- 有序数组在等值查询和范围查询场景中的性能就都非常优秀

	- 等值查询通过二分法得到，范围查询在二分法查到最小的key后，然后向右遍历直到找到第一个大于范围最大的区间值

	- 但是对于插入新的key，效率低，需要挪动插入点后面的所以key

- 使用列子

	- 有序数组索引只适用于静态存储引擎，比如你要保存的是 2017 年某个城市的所有人口信息，这类不会再修改的数据

### B树

- B树特点

	- 每个节点既保存索引，又保存数据

	- B-树和B+树最重要的一个区别就是B+树只有叶节点存放数据，其余节点用来索引，而B-树是每个索引节点都会有Data域

	- B+树中每个叶子节点有一个指向相连叶子节点的指针，而B树没有（B*树是在B+树的基础上，非根节点、非叶子节点增加了指向兄弟节点的指针）

- 使用例子

	- MongoDB使用B-树，所有节点都有Data域，只要找到指定索引就可以进行访问，无疑单次查询平均快于Mysql

### B+树

- B+树特点

	- B+树支持范围查找

		- 所有记录都是按照键值大小顺序存放在同一层的叶子节点

		- 每个叶子节点可以存放n个记录（一般一个叶子节点占用1页）

		- B+ 树的所有叶节点可以通过双向链表指针相互连接

	- 在数据中，B+树的高度一般都在2~4层，即查找某一个键值的记录最多需要2～4次io（机械硬盘1s可以至少做100次io，即2～4次io需要0.02s～0.04）

	- B+ 树的所有叶节点可以通过指针相互连接，能够减少顺序遍历时产生的额外随机 I/O

- 索引类型

	- 聚类索引和非聚类索引

		- 聚类索引（主键索引）

			- InnoDB中的主键索引就是聚类索引

			- 按照每张表的主键构建一颗B+树

			- 叶子节点中存放整张表的记录数据

			- 因为数据的逻辑顺序，聚类索引的范围查询速度快

		- 非聚类索引（普通索引）

			- 定义

				- 叶子节点除了包含键值以外，还包含主键值，之后可以根据主键值再回到主键树上去搜索其余字段

			- 种类

				- 普通索引

				- 唯一索引

			- 在普通索引上查询和在唯一索引上查询哪个效率高

				- 查询过程几乎是一样，效率没啥区别

					- 对于普通索引来说，查找到满足条件的第一个记录 (5,500) 后，需要查找下一个记录，直到碰到第一个不满足 k=5 条件的记录

					- 对于唯一索引来说，由于索引定义了唯一性，查找到第一个满足条件的记录后，就会停止继续检索

					- InnoDB 的数据是按数据页为单位来读写的，当需要读一条记录的时候，并不是将这个记录本身从磁盘读出来，而是以页为单位，将其整体读入内存。在 InnoDB 中，每个数据页的大小默认是 16KB

			- 在普通索引上查询和在唯一索引上更新哪个效率高

				- 分2种情况

					- 这个记录要更新的目标页在内存中

						- 性能基本一致

							- 对于唯一索引来说，找到要更新的位置，判断到没有冲突，插入这个值，语句执行结束

							- 对于普通索引来说，找到要更新的位置，插入这个值，语句执行结束

					- 这个记录要更新的目标页不在内存中

						- 普通索引的更新性能明显要好于唯一索引

							- 因为唯一索引的更新涉及到数据从磁盘读入内存的随机 IO操作，成本较大

								- 对于唯一索引来说，需要将数据页读入内存，判断到没有冲突，插入这个值，语句执行结束

								- 对于普通索引来说，则是将更新记录在 change buffer，语句执行就结束了

- 索引使用

	- 联合索引

		- 对表上对多个列进行索引

			- 索引内的字段顺序原则

				- 维护索引字段个数角度：如果通过调整顺序，可以少维护一个索引，那么这个顺序往往就是需要优先考虑采用的，如既有（a,b)联合索引，又有基于 a、b 各自的查询，那么此时还需要一个（b)索引

				- 维护空间角度思考：如果a字段长度比b字段大，（a,b)联合索引，a在b的前面，否则单独的a索引里，占用的空间比单独的b索引占用的空间更大

	- 覆盖索引

		- select的内容已经在普通索引里都有了

			- 不需要回表，因此覆盖索引可以减少树的搜索次数，显著提升查询性能

	- 前缀索引

		- 最左前缀原则

			- 只要满足最左前缀，就可以利用索引来加速检索。这个最左前缀可以是联合索引的最左 N 个字段，也可以是字符串索引的最左 M 个字符

		- 前缀索引对覆盖索引的影响

			- 使用前缀索引，将不能使用覆盖索引的优势了，即仍需要回表查询，因为mysql不确定前缀索引的信息是截断的还是完整的

		- 使用

			- 操作

				- 增加前缀索引

					- # 创建的 index1 索引里面，包含了每个记录的整个字符串
mysql> alter table SUser add index index1(email);
或
# 创建的 index2 索引里面，对于每个记录都是只取前 6 个字节
mysql> alter table SUser add index index2(email(6));

				- 如何更好的使用前缀索引

					- 预先有数据的情况

						- 统计索引上有多少个不同的值来判断要使用多长的前缀

							- 算出这个列上有多少个不同的值

								- select count(distinct email) as L from SUser;

							- 依次选取不同长度的前缀来看这个值，比如我们要看一下 4~7 个字节的前缀索引

								- select 
  count(distinct left(email,4)）as L4,
  count(distinct left(email,5)）as L5,
  count(distinct left(email,6)）as L6,
  count(distinct left(email,7)）as L7,
from SUser;

								- 需要预先设定一个可以接受的损失比例，比如 5%。然后，在返回的 L4~L7 中，找出不小于 L * 95% 的值，假设这里 L6、L7 都满足，就可以选择前缀长度为 6

						- 使用倒序存储（前面几位的区分度不好的情况下，可以考虑使用后面几位）

							- select field_list from t where id_card = reverse('input_id_card_string');

					- 预先没有数据

						- 使用 hash 字段 在表上再创建一个整数字段，来保存字符串的校验码，同时在这个字段上创建索引

							- alter table t add id_card_crc int unsigned, add index(id_card_crc);

					- 倒序存储和使用 hash 字段的区别

						- 都不支持范围查询

						- 从占用的额外空间来看： 倒序存储方式在主键索引上，不会消耗额外的存储空间，而 hash 字段方法需要增加一个字段，如果倒序存储使用的前缀长度较长，那么和hash存储的hash字段也差不多消耗一样的存储空间了

						- 在 CPU 消耗方面： hash计算比倒序消耗的cpu要多点

						- 从查询效率上看： 用 hash 字段方式的查询性能相对更稳定一些。hash冲突概率比前缀冲突的概率小

			- 优势

				- 使用前缀索引，定义好长度，就可以做到既节省空间，又不用额外增加太多的查询成本

	- 索引下推

		- select * from tuser where name like '张%' and age=10 and ismale=1;

			- 在 MySQL 5.6 之前，只能先找出name为张开头的ID， 然后一个个回表。到主键索引上找出数据行，再对比字段值（age、ismal）

			- 而MySQL 5.6 引入的索引下推优化（index condition pushdown)，可以在索引遍历过程中，对索引中包含的字段先做判断，直接过滤掉不满足条件的记录，减少回表次数

	- 注意点

		- 对于查询频率不高，为了效率又不能走全表搜索（性能低），为了性能又不能再建一个索引（浪费空间），如何解决

			- 一般考虑使用联合索引的方式，B+ 树这种索引结构，可以利用索引的“最左前缀”，来定位记录

- 索引维护

	- 页分裂

		- 如果要插入的一个数据所在的页已经满了，就需要申请一个新的页，然后挪动部分数据到新的页里，性能会受到影响

	- 页合并

		- 当相邻的2个页，由于删除了数据，导致利用率很低之后，会进行页合并

	- 主键一般使用自增主键，因为主健长度越小，普通索引的叶子节点就越小，普通索引占用的空间也就越小

- 注意点

	- 啥情况下适合用业务字段直接做主键？

		- 典型的 KV 场景

			- 只有一个索引

			- 该索引必须是唯一索引

	- 重建索引k

		- k是普通索引

			- 1. alter table T drop index k;
2. alter table T add index(k);

				- 重建索引是为了达到省空间的目的

		- k是主键索引

			- alter table T engine=InnoDB

				- 不能使用普通索引的重建语句

					- 不论是删除主键还是创建主键，都会将整个表重建。所以连着执行这两个语句的话，第一个语句就白做了

## 事务

### ACID特性

- 原子性(atomicity)

	- 事务系列的操作要么都成功，要么都失败

	- 通过undo log实现回滚

- 一致性(consistency)

	- 事务将数据库从一种状态转变为下一种一致的状态

	- 如果事务提交或发生回滚后，某个字段从唯一约束变成非唯一了，就破坏了一致性

- 隔离性(isolation)

	- 要求每个读写事务的对象对其他事务的操作对象能相互分离，即该事务提交前对其他事务都不可见

		- 种类

			- 读未提交

				- 一个事务还没提交时，它做的变更就能被别的事务看到

			- 读提交

				- 一个事务提交之后，它做的变更才会被其他事务看到

			- 可重复读

				- 一个事务执行过程中看到的数据，总是跟这个事务在启动时看到的数据是一致的。当然在可重复读隔离级别下，未提交变更对其他事务也是不可见的（默认级别）

			- 串行化

				- 顾名思义是对于同一行记录，“写”会加“写锁”，“读”会加“读锁”。当出现读写锁冲突的时候，后访问的事务必须等前一个事务执行完成，才能继续执行

		- 读提交的逻辑和可重复读创建视图区别

			- 在可重复读隔离级别下，只需要在事务开始的时候创建一致性视图，之后事务里的其他查询都共用这个一致性视图

			- 在读提交隔离级别下，每一个语句执行前都会重新算出一个新的视图

	- 操作

		- SHOW VARIABLES LIKE  ‘transaction%’

	- 实现原理

		- 数据库的多版本并发控制（MVCC）

			- 原理

				- 同一条记录在系统中可以存在多个版本，就是数据库的多版本并发控制（MVCC），不同时刻启动的事务再查询同一条记录时会有不同的视图

				- 每条记录在更新的时候都会同时记录一条回滚操作。记录上的最新值，通过回滚操作，都可以得到前一个状态的值

				- 每个视图中的状态值，都可以通过当前最新的值，依次执行对应的undo log得到

				- 1.png

			- 数据可见性范围

				- 一个数据版本，对于一个事务视图来说，除了自己的更新总是可见以外，有三种情况

					- 版本未提交，不可见

					- 版本已提交，但是是在视图创建后提交的，不可见

					- 版本已提交，而且是在视图创建前提交的，可见

	- 问题

		- 回滚日志的保留周期

			- 当系统里没有比这个回滚日志更早的 read-view 的时候

		- 如何避免长事物

			- 使用 set autocommit=1, 通过显式语句的方式来启动事务

			- 在 information_schema 库的 innodb_trx 这个表中查询长事务

				- select * from information_schema.innodb_trx where TIME_TO_SEC(timediff(now(),trx_started))>60

- 持久性(durability)

	- 事务一旦提交，其结果是永久性

		- 通过redo log实现（物理日志）

			- redo log特性

				- WAL技术（Write-Ahead Logging）

					- 先写日志，再写磁盘

				- redo log配置参数

					- SHOW VARIABLES LIKE '%innodb%_log%'

						- innodb_flush_log_at_trx_commit为1表示：每次事务的 redo log 都直接持久化到磁盘，建议设置为1，（同时sync_binlog设置为1时，可以保证mysql在突然断电情况下不丢数据）

						- redo log具有crash-safe能力，在mysql发生异常重启后，之前提交的记录不会丢失

							- 原因

								- 事物提交后，只会更新日志以及内存，mysql在发生crash时，可能发生数据页的丢失，而binlog里没有记录数据页的更新细节，无法补回，只有redo log里记录了物理级别的细节

						- redo log是固定大小的，比如可以配置一组4个文件（innodb_log_files_in_group参数决定），每个文件大小为1G（innodb_log_file_size 决定），如果写到末尾了，就要回到开头从头写

						- check point指的记录在redo log里没有持久化到数据库里的起始位置，write pos是追到到redo log 里最新的位置，如果write pos 追上check point，就需要停下来，将一部分记录持久化到数据库文件，使得check point往前推进，便于后续记录能够继续追加（此时可能发生数据库抖动，原本很快的sql语句，突然执行慢了，就是因为需要持久化一部分数据

	- 问题

		- changebuf 和 redolog

			- 作用

				- redo log 主要节省的是随机写磁盘的 IO 消耗（转成顺序写）

				- change buffer 主要节省的则是随机读磁盘的 IO 消耗

			- 例子

				- # k1在page1中，page1在内存
# k2在page2中，page2不在内存
select * from t where k in (k1, k2)

					- 执行过程

						- Page 1 在内存中，直接更新内存

						- Page 2 没有在内存中，就在内存的 change buffer 区域，记录下“我要往 Page 2 插入一行”这个信息

						- 将上述两个动作记入 redo log 中

						- 因此执行这条更新语句的成本很低，就是写了两处内存，然后写了一处磁盘（两次操作合在一起写了一次磁盘），而且还是顺序写的

			- changebuf使用场景

				- change buffer 的主要目的就是将记录的变更动作缓存下来，所以在一个数据页做 merge 之前，change buffer 记录的变更越多（也就是这个页面上要更新的次数越多），收益就越大

					- 对于写多读少的业务来说，页面在写完以后马上被访问到的概率比较小，此时 change buffer 的使用效果最好。这种业务模型常见的就是账单类、日志类的系统

					- 如果一个业务是写完后，立马就要读取，会立即触发merge过程，这样随机访问 IO 的次数不会减少，反而增加了 change buffer 的维护代价

			- changebuf merge过程

				- 从磁盘读入数据页到内存（老版本的数据页）

				- 从 change buffer 里找出这个数据页的 change buffer 记录 (可能有多个），依次应用，得到新版数据页

				- 写 redo log。这个 redo log 包含了数据的变更和 change buffer 的变更

				- 到这里 merge 过程就结束了。此时，数据页和内存中 change buffer 对应的磁盘位置都还没有修改，属于脏页，之后各自刷回自己的物理数据，就是另外一个过程了

			- change buffer持久化触发情况

				- 数据库空闲时，后台有线程定时持久化

				- 数据库缓冲池不够用时

				- 数据正常关闭时

				- redo log 写满时

			- 如果某次写入使用了 change buffer 机制，之后主机异常重启，是否会丢失 change buffer 和数据

				- 不会丢失

					- 虽然是只更新内存，但是在事务提交的时候，我们把 change buffer 的操作也记录到 redo log 里了，所以崩溃恢复的时候，change buffer 也能找回来

### 事物的启动方式

- 显式启动事务语句， begin 或 start transaction。配套的提交语句是 commit，回滚语句是 rollback

	- begin/start transaction 命令并不是一个事务的起点，在执行到它们之后的第一个操作 InnoDB 表的语句，事务才真正启动（一致性视图是在执行第一个快照读语句时创建的）

	- 如果你想要马上启动一个事务，可以使用 start transaction with consistent snapshot 这个命令（一致性视图是在执行 start transaction with consistent snapshot 时创建的）

- set autocommit=0，这个命令会将这个线程的自动提交关掉。意味着如果你只执行一个 select 语句，这个事务就启动了，而且并不会自动提交。这个事务持续存在直到你主动执行 commit 或 rollback 语句，或者断开连接

### 问题

- 系统里面应该避免长事务， 什么方案来避免出现或者处理这种情况呢

	- 1. 从应用开发端来看

		- 确认是否使用了 set autocommit=0。这个确认工作可以在测试环境中开展，把 MySQL 的 general_log 开起来，然后随便跑一个业务逻辑，通过 general_log 的日志来确认。一般框架如果会设置这个值，也就会提供参数来控制行为，你的目标就是把它改成 1

		- 确认是否有不必要的只读事务。有些框架会习惯不管什么语句先用 begin/commit 框起来。我见过有些是业务并没有这个需要，但是也把好几个 select 语句放到了事务中。这种只读事务可以去掉

		- 业务连接数据库的时候，根据业务本身的预估，通过 SET MAX_EXECUTION_TIME 命令，来控制每个语句执行的最长时间，避免单个语句意外执行太长时间。

	- 2. 从数据库端来看

		- 监控 information_schema.Innodb_trx 表，设置长事务阈值，超过就报警 / 或者 kill

		- Percona 的 pt-kill 这个工具不错，推荐使用

		- 在业务功能测试阶段要求输出所有的 general_log，分析日志行为提前发现问题

		- 如果使用的是 MySQL 5.6 或者更新版本，把 innodb_undo_tablespaces 设置成 2（或更大的值）。如果真的出现大事务导致回滚段过大，这样设置后清理起来更方便

### redolog与binlog日志

- 二阶段提交

	-  

		- 发生crash的情况

			- 时刻t1 crash redo log 和 binlog都在内存中，所以本次事物的相关操作都消失

			- 时刻t2 crash redo log已经写到磁盘，但是binlog没有写到磁盘。mysql重启后，读取磁盘中的redo log，但是由于redo log处于prepare状态，就要判断binlog是否完整，如果完整，就提交事务，否则回滚

			- 时刻t3 crash redolog和binlog都在磁盘里，此时从redolog恢复数据即可

		- 崩溃恢复时的判断规则

			- 如果 redo log 里面的事务是完整的，也就是已经有了 commit 标识，则直接提交

			- 如果 redo log 里面的事务只有完整的 prepare，则判断对应的事务 binlog 是否存在并完整

				- 如果是，则提交事务

				- 否则，回滚事务

- redo log与 binlog关联

	- 它们有一个共同的数据字段，叫 XID

	- 崩溃恢复的时候，会按顺序扫描 redo log：如果碰到既有 prepare、又有 commit 的 redo log，就直接提交

	- 如果碰到只有 parepare、而没有 commit 的 redo log，就拿着 XID 去 binlog 找对应的事务

## 复制

### bin log（逻辑日志）

- bin log特性

	- binlog_format为statement，表示记录的是sql语句，row表示记录的是原始数据（同步数据更好，但是日志文件较大），mix表示sql语句于原始数据结合

	- binlog 日志只能用于归档，用于主从同步

	- sync_binlog为：表示每次事务的 binlog 都持久化到磁盘。这个参数我也建议你设置成 1，这样可以保证 MySQL 异常重启之后 binlog 不丢失

- binlog怎么判断完整性

	- statement 格式的 binlog，最后会有 COMMIT

	- row 格式的 binlog，最后会有一个 XID event

	- 对于 binlog 日志由于磁盘原因，可能会在日志中间出错的情况，MySQL 可以通过校验 checksum 的结果来发现

		- 通过binlog-checksum 参数，用来验证 binlog 内容的正确性

- 查看binlog日志

	- SHOW BINARY LOGS

	- SHOW BINLOG EVENT IN `mysql-bin.000002`

## 性能调优

### 问题

- 短连接风暴

- 查询缓存

	- 大多数情况下我会建议你不要使用查询缓存，为什么呢？因为查询缓存往往弊大于利

		- 查询缓存的失效非常频繁，只要有对一个表的更新，这个表上所有的查询缓存都会被清空。对于更新压力大的数据库来说，查询缓存的命中率会非常低

		- 将参数 query_cache_type 设置成 DEMAND，这样对于默认的 SQL 语句都不使用查询缓存

		- 对于你确定要使用查询缓存的语句，可以用 SQL_CACHE 显式指定

			- # MySQL 8.0 已经去掉了缓存功能
mysql> select SQL_CACHE * from T where ID=10；

- MySQL 偶尔“抖”一下的那个瞬间，可能是在刷脏页

	- 可能原因

		- InnoDB 的 redo log 写满了

			- 这种情况应该尽量避免，因为出现这种情况的时候，整个系统就不能再接受更新了，所有的更新都必须堵住

		- 系统内存不足

			- 当需要新的内存页，而内存不够用的时候，就要淘汰一些数据页，空出内存给别的数据页使用。如果淘汰的是“脏页”，就要先将脏页写到磁盘

		- MySQL 认为系统“空闲”的时候

			- 因为系统空闲，因此对性能不关心

			- MySQL 认为系统“空闲”的时候，就会刷脏页

		- MySQL 正常关闭的情况

			- 因为要关闭mysql了，所以对性能不关心

			- MySQL 正常关闭时，MySQL 会把内存的脏页都 flush 到磁盘上，这样下次 MySQL 启动的时候，就可以直接从磁盘上读数据，启动速度会很快

	- 解决方法

		- InnoDB 刷脏页的控制策略

			- 要正确地告诉 InnoDB 所在主机的 IO 能力，这样 InnoDB 才能知道需要全力刷脏页的时候，可以刷多快

				- 一般将innodb_io_capacity参数设置成磁盘的 IOPS

				- 如何设置的太低：导致MySQL 的写入速度很慢，TPS 很低：所以刷脏页刷得特别慢（导致内存脏页太多，以及redo log 写满），甚至比脏页生成的速度还慢，这样就造成了脏页累积，影响了查询和更新性能

				- # 测试磁盘随机读写
 fio -filename=$filename -direct=1 -iodepth 1 -thread -rw=randrw -ioengine=psync -bs=16k -size=500M -numjobs=10 -runtime=10 -group_reporting -name=mytest

			- InnoDB 的刷盘速度就是要参考这两个因素

				- 脏页比例

					- 参数 innodb_max_dirty_pages_pct 是脏页比例上限，默认值是 75%

				- redo log 写盘速度

			- 刷脏页是否连带邻居数据页

				- MySQL 中的一个机制：在准备刷一个脏页的时候，如果这个数据页旁边的数据页刚好是脏页，就会把这个“邻居”也带着一起刷掉；而且这个把“邻居”拖下水的逻辑还可以继续蔓延

				- innodb_flush_neighbors参数为1，则会连带邻居数据页一起刷，为0则不会

				- 机械硬盘建议设置innodb_flush_neighbors参数为1，SSD可以设置为0

				- 在 MySQL 8.0 中，innodb_flush_neighbors 参数的默认值已经是 0 了

## 部署运维

### 生产环境MySQL添加或修改字段的方法

- 直接添加

	- 如果该表读写不频繁，数据量较小（通常1G以内或百万以内），直接添加即可（可以了解一下online ddl的知识）

- 使用pt_osc添加

	- 如果表较大 但是读写不是太大，且想尽量不影响原表的读写，可以用percona tools进行添加，相当于新建一张添加了字段的新表，再将原表的数据复制到新表中，复制历史数据期间的数据也会同步至新表，最后删除原表，将新表重命名为原表表名，实现字段添加

- 先在从库添加 再进行主从切换

	- 如果一张表数据量大且是热表（读写特别频繁），则可以考虑先在从库添加，再进行主从切换，切换后再将其他几个节点上添加字段

## 其他

### 问题

- 为什么表数据删掉一半，表文件大小不变

	- 原因

		- 如果表数据存放在共享空间里，删除一个表，空间也是不会回收的

			- innodb_file_per_table 参数OFF

				- 表的数据放在系统共享表空间，也就是跟数据字典放在一起

			- innodb_file_per_table 参数ON

				- 每个 InnoDB 表数据存储在一个以 .ibd 为后缀的文件中

	- 数据删除流程

		- delete 命令其实只是把记录的位置，或者数据页标记为了“可复用”，但磁盘文件的大小是不会变的

		- 如果单个记录的位置可以复用，那么需要插入主键ID大小合适的记录，才能使用该位置

		- 如果整个页都可以复用，那么插入新的数据时，都可能复用该页

		- 增删改都会导致数据页空洞，如相连的数据页利用率太低，那么会将数据合并为一个数据页中，另外一个页标记为复用，空间大小不会减小

	- 减小空洞，收缩表空间

		- 命令

			- # 重建表
alter table A engine=InnoDB

		- 流程

			- MySQL 5.6 版本开始引入的 Online DDL（重建表的过程中，允许对备份表 做增删改操作）

				- 主要流程

					- 建立一个临时文件，扫描表 A 主键的所有数据页

					- 用数据页中表 A 的记录生成 B+ 树，存储到临时文件中

					- 生成临时文件的过程中，将所有对 A 的操作记录在一个日志文件（row log）中，对应的是图中 state2 的状态

					- 临时文件生成后，将日志文件中的操作应用到临时文件，得到一个逻辑数据上与表 A 相同的数据文件，对应的就是图中 state3 的状态

					- 用临时文件替换表 A 的数据文件

					- 2.png

				- 说明

					- alter 语句在启动的时候需要获取 MDL 写锁，但是这个写锁在真正拷贝数据之前就退化成读锁了

					- 为什么要退化呢？为了实现 Online，MDL 读锁不会阻塞增删改操作

					- 那为什么不干脆直接解锁呢？为了保护自己，禁止其他线程对这个表同时做 DDL

					- 上述的这些重建方法都会扫描原表数据和构建临时文件。对于很大的表来说，这个操作是很消耗 IO 和 CPU 资源的。因此，如果是线上服务，要很小心地控制操作时间。如果想要比较安全的操作的话，推荐使用 GitHub 开源的 gh-ost 来做

		- optimize table、analyze table 和 alter table区别

			- 从 MySQL 5.6 版本开始，alter table t engine = InnoDB（也就是 recreate）默认的就是上面图流程

			- analyze table t 其实不是重建表，只是对表的索引信息做重新统计，没有修改数据，这个过程中加了 MDL 读锁

			- optimize table t 等于 recreate+analyze

- 在执行重建表后，表空间反而变大了，是啥原因

	- 命令

		- # 重建表
alter table A engine=InnoDB

	- 可能原因

		- 原因

			- 这个表，本身就已经没有空洞的了，比如说刚刚做过一次重建表操作

			- 在 DDL 期间，如果刚好有外部的 DML 在执行，这期间可能会引入一些新的空洞

		- 注意

			- 在重建表的时候，InnoDB 不会把整张表占满，每个页留了 1/16 给后续的更新用。也就是说，其实重建表之后不是“最”紧凑的

				- 可能过程

					- 将表 t 重建一次

					- 插入一部分数据，但是插入的这些数据，用掉了一部分的预留空间

					- 这种情况下，再重建一次表 t，就可能会出现问题中的现象

## 命令操作

### 字符编码

- 查看数据库编码

	- SHOW CREATE DATABASE ti_data_center;

### 连接数

- 查看当前最大连接数

	- show variables like '%max_connections%';

- 查看连接最多的ip

	- select SUBSTRING_INDEX(host,':',1) as ip , count(*) as count from information_schema.processlist 
group by ip order by count desc;

- 重新设置最大连接数

	- set global max_connections=1000;

	- vi /etc/my.cnf

		- set-variable=max_user_connections=30 这个就是单用户的连接数

		- set-variable=max_connections=800 这个是全局的限制连接数

- 当前连接数/当前并发数

	- show status like 'Threads%';

		- threads_connected表示当前连接数

		- threads_running表示当前并发数

- 显示前100条连接信息

	- show processlist;

- 显示全部连接信息

	- show full processlist;

- 断开连接

	- kill id

### 预处理语句

- 同时维护的预处理语句的最大数

	- SHOW VARIABLES LIKE 'max_prepared_stmt_count';

- 当前占用的Prepared_stmt_count

	- show global status like '%stmt%';

### 分区

- 增加分区

	-  ALTER TABLE task ADD PARTITION (PARTITION p20210115 VALUES LESS THAN (TO_SECONDS ('2021-01-16 00:00:00')));

### 事务

- 查看事务状态

	- # 纵向查看\G
select * from information_schema.innodb_trx\G;

		- rx_state

			- 事务状态，一般为RUNNING

		- trx_started

			- 事务执行的起始时间，若时间较长，则要分析该事务是否合理

		- trx_mysql_thread_id

			- MySQL的线程ID，用于kill

		- trx_query

			- 事务中的sql

### 修改表结构

- 增加字段

	- alter table data_source_export add username varchar(64) NOT NULL DEFAULT '';

- 修改字段类型

	- alter table data_source_export modify username varchar(64) NOT NULL DEFAULT '';

- 修改字段名称

	- alter table task change updated_at update_time  datetime NOT NULL DEFAULT CURRENT_TIMESTAMP;

- 删除字段

	- ALTER TABLE task drop updated_at;

- 增加索引

	- alter table data_source_export add index idx_username(username);

- 删除索引

	- ALTER TABLE table_name DROP INDEX index_name;

### 主备

- 查询主库

	- select * from performance_schema.replication_group_members;

### 日志

- 慢日志

	- 查询慢查询日志是否开启

		-  show variables like '%slow%';

	- 开启慢查询

		- set global slow_query_log=ON;

	- 日志路径

		- SHOW VARIABLES LIKE "slow_query_log_file";

	- 慢查询的时间阈值

		- SHOW VARIABLES LIKE "long_query_time%"

	- 设置慢查询时间阈值

		- set global long_query_time=1;

	- 查询日志内容

		- SELECT * from mysql.general_log WHERE thread_id<>'19' ORDER BY event_time DESC;

		- SELECT * from mysql.slow_log  WHERE thread_id<>'19' ORDER BY event_time DESC;

- 设置log_output 类型

	- set global log_output=’table’;

	- set global log_output=’FILE’;

- 日志存储方式

	- SHOW VARIABLES LIKE "%log_output%";

### 备份/恢复

- 备份

	- 从数据库导出数据

		- # mysqldump -h 127.0.0.1 -P(大写)端口 -u用户名 -p密码 数据库名>bak.sql(路劲)

mysqldump -h 132.72.192.432 -P3307 -uroot -p8888 htgl > bak.sql;

	- 导出数据和表结构

		- 将特定数据库特定表中的数据和表格结构和数据全部返回

			- mysqldump --u  b_user -h 101.3.20.33 -p'H_password'  -P3306 database_di up_subjects > 0101_0630_up_subjects.sql

	- 导出表结构但不导出表数据

		- 只返回特定数据库特定表格的表格结构，不返回数据,添加“-d”命令参数

			- mysqldump --u  b_user -h 101.3.20.33 -p'H_password'  -P3306 -d database_di up_subjects > 0101_0630_up_subjects.sql

	- 导出表结构和满足指定条件的表数据

		- 只返回特定数据库中特定表的表格结构和满足特定条件的数据

			- mysqldump --u  b_user -h 101.3.20.33 -p'H_password'  -P3306 database_di up_subjects --where=" ctime>'2017-01-01' and ctime<'2017-06-30'" > 0101_0630_up_subjects.sql

	- 导出数据却不导出表结构

		- 只返回特定数据库中特定表格的数据，不返回表格结构，添加“-t”命令参数

			- mysqldump --u  b_user -h 101.3.20.33 -p'H_password' -t -P3306 database_di up_subjects  >0101_0630_up_subjects.sql

	- 导出特定数据库的所有表格的表结构及其数据，添加“--databases ”命令参数

		- mysqldump  --u  b_user -h 101.3.20.33 -p'H_password' -P3306 --databases test  > all_database.sql

- 恢复导入数据库数据

	- 系统命令行

		- 格式：mysql -h 127.0.0.1 -P(大写)端口 -u用户名 -p密码 数据库名 < XX.sql(路劲) 

mysql -uusername -ppassword db1 <tb1tb2.sql

	- mysql命令行

		- user db1;
source tb1_tb2.sql;

	- 恢复整个数据库的方法

		- mysql -u  b_user -h 101.3.20.33 -p'H_password' -P3306   < all_database.sql

### 查询

- 查询最后一条记录

	- select * from table order by id DESC limit 1

- 查询结果保存csv文件

	- [SELECT order_id,product_name,qty
INTO OUTFILE '/var/lib/mysql-files/orders.csv'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
FROM orders
WHERE foo = 'bar';](https://stackoverflow.com/questions/356578/how-can-i-output-mysql-query-results-in-csv-format)

- [limit使用注意点](https://bugs.mysql.com/bug.php?id=69732)

	- 一定要加order by 唯一索引

