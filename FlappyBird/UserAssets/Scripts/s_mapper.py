queue = []
__id__ = 'mapper'
last_pos = Vector3()
map_lookahead_offset = 100
max_displacement_allowed = 8


def Start():
    Camera.clearColor = Vector3(0, 0, .2)


def Update():
    if last_pos.x <= Camera.position.x + map_lookahead_offset:
        height = random.randrange(-2, 2)
        upper_wall = instantiate_script('pipe')
        lower_wall = instantiate_script('pipe')

        upper_wall.transform.position = Vector3(last_pos.x, height - 4, -1)
        lower_wall.transform.position = Vector3(last_pos.x, height + 4, -1)
        lower_wall.transform.scale.y = -1
        queue.append([last_pos.x, height])

        last_pos.x += max_displacement_allowed
