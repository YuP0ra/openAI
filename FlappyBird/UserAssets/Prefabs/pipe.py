transform = Transform2D()
pipe = SpriteRenderer('pipe.png')
collider = BoxCollider2D(2, 3.7, transform, __id__, collider_tag='pipe')


def Render():
    transform.applyTransformation()
    pipe.render()

