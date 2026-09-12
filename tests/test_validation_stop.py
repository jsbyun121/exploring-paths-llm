import unittest
from RL2.utils.validation_stop import record_validation

class ValidationStopTests(unittest.TestCase):
    def test_exact_rule_and_ties(self):
        h=[]
        for step,score,expected in [(100,.8,False),(120,.9,False),(140,.85,False),(160,.85,False),(180,.84,True)]:
            h,stop,_=record_validation(h,step,score);self.assertEqual(stop,expected)
    def test_resume_replaces_future_and_no_duplicate(self):
        h=[{'step':100,'accuracy':.8},{'step':120,'accuracy':.9},{'step':140,'accuracy':.7}]
        h,stop,_=record_validation(h,120,.88)
        self.assertFalse(stop);self.assertEqual(len(h),2)
    def test_best_and_invalid(self):
        h,_,_=record_validation([],100,.9)
        _,_,best=record_validation(h,120,.89);self.assertFalse(best)
        with self.assertRaises(ValueError):record_validation(h,120,float('nan'))
