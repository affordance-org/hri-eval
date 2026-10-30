"""The catalog that run.py --list prints, and that docs/catalog.md shows: what is available today and what is on
request (not built yet). Every on-request item has a link that opens a GitHub issue form with the title and the
one-line field already filled in."""
from urllib.parse import quote

REPO = 'https://github.com/affordance-org/hri-eval'
AVAILABLE, ON_REQUEST = 'available', 'on request'

# issue form -> (title prefix, id of the one-line field the link fills in)
FORMS = {'request_scenario.yml': ('[Scenario] ', 'scenario'), 'request_robot.yml': ('[Robot] ', 'robot_name'),
         'request_feature.yml': ('[Feature] ', 'feature'), 'hosted_eval.yml': ('[Hosted eval] ', None),
         'submit_result.yml': ('[Result] ', None), 'bug.yml': ('[Bug] ', None)}


def request_url(template, text=''):
    """Link to a new issue from `template` with the title (and, for a named item, its one-line field) filled in."""
    prefix, field = FORMS[template]
    url = f'{REPO}/issues/new?template={template}&title={quote(prefix + text)}'
    return url + (f'&{field}={quote(text)}' if text and field else '')


def scenario_request(name):
    return request_url('request_scenario.yml', name)


def feature_request(name):
    return request_url('request_feature.yml', name)


def robot_request(name):
    return request_url('request_robot.yml', name)


# The five scores: (report.json key, column name, one line, checker gate). Definitions: docs/scoring.md.
SCORES = [('no_fall', 'no fall', 'stays up, touches only what it is allowed to touch, and the episode ends normally',
           'G2_no_fall'),
          ('pickup', 'pickup', 'lifts the bag 5 cm, held only by the finger pads, for 0.5 s', 'G3_pickup'),
          ('carry', 'carry', 'carries it 0.4 m while walking 0.3 m', 'G4_carry'),
          ('handover', 'handover', 'the person alone holds the bag for the last 1.0 s or more', 'G5_handover'),
          ('pace', 'pace', "handover complete within 3 x the human's 4.4 s", 'G6_pace')]

SCENARIOS = [('point-fetch', 'a recorded person points at a backpack on a low table; the robot fetches it and hands '
                             'it over (docs/scenario-point-fetch.md)')]


def A(name, detail=''):
    return (name, AVAILABLE, detail)


def R(name, url):
    return (name, ON_REQUEST, url)


def sections():
    """[(section title, [(item, status, detail or request link)])]"""
    return [
        ('Scenarios (--scenario NAME)',
         [A(n, d) for n, d in SCENARIOS] +
         [R(n, scenario_request(n)) for n in ['Handover to a seated person', 'Passing a person in a narrow space',
                                              'Tidying a table next to a working person', 'Sit-to-stand help',
                                              'Carrying with a person']]),
        ('People and rooms',
         [A('One recorded person in one room', 'recorded with cameras in a real room, replayed in the room rebuilt in '
                                               'MuJoCo; hands, forearms and body are colliders fitted to the skin')] +
         [R(n, feature_request(n)) for n in ['Reactive humans that respond to the robot', 'More recorded people and rooms',
                                             'Capturing your own room and people']]),
        ('Robots',
         [A('g1-umi', 'Unitree G1, 29 joints, two UMI parallel grippers (35 N per finger)')] +
         [R(n, robot_request(n)) for n in ['G1 with Dex3 hands', 'Unitree H1', 'Fourier GR-1',
                                           'Mobile manipulators such as Hello Robot Stretch', 'Your own URDF or MJCF']]),
        ('Policy interfaces (--policy MODULE:CLASS or path/to/file.py:CLASS)',
         [A('Joint level', 'reset(ctx), policy_step(ctx) -> episode.ControlOutput (example: examples/stand_still.py)'),
          A('Command level', 'interfaces.CommandPolicy.act(obs) -> interfaces.Command: base velocity to GR00T, 14 arm '
                             'joint targets, 2 grippers (example: examples/walk_to_table.py)'),
          A('GR00T walking helper', 'episode.Gr00tLocomotion and episode.compose_gr00t, for joint level policies'),
          A('--params', 'JSON or a .json file, passed to your constructor as keywords')] +
         [R(n, feature_request(n)) for n in ['End-effector pose action interface',
                                             'Event logging for command level policies', 'LeRobot policy adapter',
                                             'ROS 2 bridge', 'Remote policy server (websocket or gRPC)']]),
        ('Observations',
         [A('Robot proprioception', 'joint positions and velocities, pelvis pose and velocities, finger positions'),
          A('IMU sensors', 'pelvis and torso gyro, accelerometer, orientation (joint level: ctx.scratch().sensordata)'),
          A('Person keypoints', '20 points at the current time, ideal (obs["person"]; joint level: ctx.recipient)'),
          A('Object poses', 'true backpack pose, the fixed table objects (obs["objects"]; joint level: ctx.body)'),
          A('Full simulator state', 'read only (joint level: ctx.qpos, ctx.d, ctx.scratch())')] +
         [R(n, feature_request(n)) for n in ['Rendered RGB and depth camera observations',
                                             'Noisy or occluded person keypoints', 'Force-torque and tactile sensors']]),
        ('Scores (report.json; the checker re-simulates every trial)',
         [A('valid', "the checker's G0 invariants and G1 motor replay pass")] +
         [A(k, d) for k, _, d, _ in SCORES] +
         [A('Checker gates G0 to G9', 'all ten, with their measurements, in gates.json'),
          A('Minimum robot to person distance', 'per episode, a measurement, not a score'),
          R('Comfort and proxemics scores', feature_request('Comfort and proxemics scores'))]),
        ('Variations and batches (--episodes N --seed S runs seeds S..S+N-1; seed 0 is the nominal recorded episode)',
         [A('Robot start offset', 'dx, dy within +-0.10 m, yaw within +-10 deg'),
          A('Backpack start offset', 'dx, dy on the table within +-0.05 m'),
          A('Person playback speed', 'factor within 0.9 to 1.1'),
          A('--jobs J', 'episodes in parallel processes'),
          R('More variation types', feature_request('More variation types'))]),
        ('Tooling',
         [A('run.py', 'run, score and report in one command'),
          A('report.json and report.md', 'per run, in the --out folder'),
          A('Trial folders', 'full trajectory, logs, events and disclosures per episode (docs/trial-format.md)'),
          A('hri_eval/check_gates.py', 're-score any trial folder'),
          A('run.py --list', 'this list'),
          R('Trial viewer and video export', feature_request('Trial viewer and video export'))]),
        ('Evaluation services',
         [A('Leaderboard table in this repository', 'updated by hand from submitted results (docs/leaderboard.md)'),
          R('Hosted evaluation (send us a checkpoint, we run it)', request_url('hosted_eval.yml')),
          R('Public leaderboard service', feature_request('Public leaderboard service')),
          R('Real-robot replay in the same room', feature_request('Real-robot replay in the same room'))]),
    ]


def on_request_items():
    return [(n, d) for _, rows in sections() for n, st, d in rows if st == ON_REQUEST]


def print_list():
    secs = sections()
    w = max(len(n) for _, rows in secs for n, _, _ in rows) + 2
    for title, rows in secs:
        print(title)
        for n, st, d in rows:
            print(f'  {n:<{w}}{st:<12} {d}')
        print()
    print(f"Something else? {request_url('request_feature.yml')}")
