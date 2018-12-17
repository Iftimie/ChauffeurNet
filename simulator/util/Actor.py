
class Actor:

    def __init__(self):
        pass

    def render(self, image, C):
        """
        :param image: image on which this actor will be renderd on
        :param C:     camera matrix
        :return:      image with this object renderd
        """
        pass

    def set_transform(self, x=0, y=0, z=0, roll=0, yaw=0, pitch=0):
        """
        values in world coordinates

        X axis positive to the right, negative to the right
        Y axis positive up, negative down
        Z axis positive forward, negative backwards
        :param roll:  angle in degrees around Z axis
        :param yaw:   angle in degrees around Y axis
        :param pitch: angle in degrees around X axis
        """
        pass