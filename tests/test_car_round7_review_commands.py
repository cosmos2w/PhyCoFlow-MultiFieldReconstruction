from pathlib import Path

from scripts.posttraining.run_car_round7_review import ARMS, commands


def test_review_commands_are_bounded_distinct_and_use_frozen_code():
    plans = [commands('/frozen', '/env/python', '/data/campaign', '/data/runtime.json',
                      '/data/campaign/assessment/attempt', arm, 'cuda:0', 20000, 4000)
             for arm in ARMS]
    for arm, plan in zip(ARMS, plans):
        training = plan['training']
        banks = plan['bank_generation']
        assert training[1] == '/frozen/scripts/posttraining/lease_car_round7.py'
        assert 'learning' in training and 'assessment' in banks
        assert '/frozen/scripts/posttraining/run_car_round7.py' in training
        assert training[training.index('--arm') + 1] == arm
        assert training[training.index('--result') + 1] == plan['training_result']
        assert Path(plan['training_result']).name == 'training_result.json'
        assert '/frozen/scripts/posttraining/run_car_round7_review.py' in banks
        assert '_generate-four' in banks
    assert plans[0]['training_result'] != plans[1]['training_result']
