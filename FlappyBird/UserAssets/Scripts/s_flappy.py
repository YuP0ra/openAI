__id__ = 'bird'
transform = Transform2D()
pipe = SpriteRenderer('bird.png')
collider = BoxCollider2D(.5, .5, transform, __id__, collider_tag='bird')


def Start():
    global queue, model, collision, counter, state

    counter = 0
    state = None
    collision = False
    queue = get_script('mapper').queue
    model = get_script('neural_network')
    transform.scale = Vector3(.25, .25, .25)
    collider.on_collision_trigger(OnCollision)


def Render():
    transform.applyTransformation()
    pipe.render()


def Update():
    global model, counter, state

    if state is None:
        state = reset()

    if Time.fixedTime > 4:
        predictions = model.get_quality(state.reshape(-1, 3))
        chosen_action = np.argmax(predictions, axis=1)[0]
        next_state, reward, done = step(chosen_action)
        model.append_memory(state, chosen_action, reward, predictions)

        if done:
            train()
            state = reset()

        state = next_state

    counter = counter + 1


def reset():
    global pipe_index, speed

    transform.position = Vector3(-6, 0, -2)
    Camera.position = Vector3(0, 0, -10)
    speed = Vector3(2, 0, 0)
    pipe_index = 0

    return np.asarray([0, 0, 0])


def step(action):
    global speed, queue, collision, pipe_index

    # Perform the action and update the env.
    if action == 0:
        speed.y = 5

    speed += Vector3(0, -10, 0) * Time.deltaTime
    transform.position += speed * Time.deltaTime
    Camera.position.x = lerp(Camera.position.x , transform.position.x + 3, Time.deltaTime)

    # Calculate the reward and check for game termination.
    reward, done = -0.1 * abs(transform.position.y), False
    if abs(transform.position.y) > 6 or collision:
        collision = False
        done = True
        reward = -1

    # check for success pipe passing
    if math.ceil(transform.position.x / 8) > pipe_index:
        pipe_index += 1
        reward = 3

    next_state = np.asarray([transform.position.x - queue[pipe_index][0],
                            transform.position.y - queue[pipe_index][1],
                            speed.y])

    return next_state, reward, done


def train():
    global model
    model.discount_buffer()
    model.train_patch()
    model.reset_memory()


def OnCollision(hits):
    global playing, collision
    for hit in hits:
        if hit == 'pipe':
            collision = True
            castEvent('died', None)
