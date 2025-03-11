#================================================================
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../")) 

from ipsim import *

import pytest

#================================================================
def test_value_sources_1():
    v = ValueSource(value=1)
    assert v() == 1

def test_value_sources_2():
    v = ValueSource(value=5.5)
    assert v() == 5.5

def test_value_sources_3():
    v1 = ValueSource(value = 3)
    v2 = ValueSource(source=v1)
    assert v1() ==v2() == 3

def test_model_1():
    pmodel = ProcessModel("test",dt=0.01)
    pmodel.add_node(ProcessInputNode("TestReservoir1", {"Temperature":300}))
    assert pmodel.next_state( ("TestReservoir1", ))["TestReservoir1"]["Temperature"] == 300

#----------------------------------------------------------------
class DublingSensor(Sensor):
        def __init__(self, name, input_name):
            super().__init__(name,input_name)

        def evaluate(self):
            i = self.inputs()
            for key in i:
                self.set_result(key, i[key]()*2)

#----------------------------------------------------------------
def test_model_2():
    processModel = ProcessModel("test",dt=0.01)
    processModel.add_node(ProcessInputNode("Reservoir1_Coolant",{"Temperature":300}))
    processModel.add_node(Sensor("R1CTS1","Temperature"))
    processModel.add_node(DublingSensor("R1CTS2","Temperature"))
    processModel.bond_nodes("R1CTS1","Temperature", "Reservoir1_Coolant", "Temperature")
    processModel.bond_nodes("R1CTS2","Temperature", "Reservoir1_Coolant", "Temperature")

    for i in range(5):
        state = processModel.next_state(("R1CTS1","R1CTS2"))
        assert state["R1CTS1"]["Temperature"] == 300
        assert state["R1CTS2"]["Temperature"] == 600

def test_model_3():
    processModel = ProcessModel("test0",dt=0.01)
    processModel.add_node(ProcessInputNode("Reservoir1_Coolant",{"Temperature":300}))
    processModel.add_node(Sensor("R1CTS1","Temperature"))
    processModel.add_node(DublingSensor("R1CTS2","Temperature"))
    processModel.bond_nodes("R1CTS1","Temperature", "Reservoir1_Coolant", "Temperature")
    processModel.bond_nodes("R1CTS2","Temperature", "Reservoir1_Coolant", "Temperature")

    for i in range(10):
        inode = processModel.nodes()["Reservoir1_Coolant"]
        inode.change_value("Temperature",inode.value("Temperature")+1)
        state = processModel.next_state(("R1CTS1","R1CTS2"))
        assert state["R1CTS1"]["Temperature"] == inode.value("Temperature")
        assert state["R1CTS2"]["Temperature"] == inode.value("Temperature")*2