class Annotations:

    def __init__(self, offset=(0.0, 0.0, 0.0)):
        # Annotations的私有属性types，存储对象的类型：presynaptic_site/postsynaptic_site
        self.__types = {}
        # Annotations的私有属性locations，存储对象的空间位置
        self.__locations = {}
        # Annotations的私有属性locations，存储对象的空间位置
        self.comments = {}
        # 存放突触前后对象的目标对（突触前膜id，突触后膜id）
        self.pre_post_partners = []
        self.offset = offset

    def __check(self, id):
        # 检查数据库中是否存在该id对象的注释
        if not id in self.__types.keys():
            raise "there is no annotation with id " + str(id)

    def add_annotation(self, id, type, location):
        """Add a new annotation.

        Parameters
        ----------

            id: int
                The ID of the new annotation.

            type: string
                A string denoting the type of the annotation. Use 
                "presynaptic_site" or "postsynaptic_site" for pre- and 
                post-synaptic annotations, respectively.

            location: tuple, float
                The location of the annotation, relative to the offset.
        """
        # 增加对应id的键值对
        # encode() 方法以指定的编码格式编码字符串，默认编码为 ‘utf-8’，编码为bytes对象。
        # 对应的解码方法：bytes decode() 方法，bytes:字节。该方法返回编码后的字符串，它是一个 bytes 对象
        # types:对象类型"presynaptic_site" or "postsynaptic_site"
        self.__types[id] = type.encode('utf8')
        self.__locations[id] = location

    def add_comment(self, id, comment):
        """Add a comment to an annotation.
        """
        # 检查id是否存在
        self.__check(id)
        self.comments[id] = comment.encode('utf8')

    def set_pre_post_partners(self, pre_id, post_id):
        """Mark two annotations as pre- and post-synaptic partners.
        """

        self.__check(pre_id)
        self.__check(post_id)
        self.pre_post_partners.append((pre_id, post_id))

    def ids(self):
        """Get the ids of all annotations.
        """
        # 返回types的键，即id标号
        # python3中使用dict.keys()返回的不在是list类型了，也不支持索引
        return list(self.__types.keys())

    def types(self):
        """Get the types of all annotations.
        """
        # 返回types的键，即对象的类型
        # python3中使用dict.keys()返回的不在是list类型了，也不支持索引
        return list(self.__types.values())

    def locations(self):
        """Get the locations of all annotations. Locations are in world units, 
        relative to the offset.
        """

        return list(self.__locations.values())

    def get_annotation(self, id):
        """Get the type and location of an annotation by its id.
        """

        self.__check(id)
        return (self.__types[id], self.__locations[id])
