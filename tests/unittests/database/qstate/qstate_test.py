from exojax.database.molinfo  import m_transition_state

def test_m_transition_state():
    # See HITRAN 2020 paper
    for i in range(1,100):
        assert m_transition_state(i,1) == i+1 # R-branch 
        assert m_transition_state(i,0) == i # Q-branch 
        assert m_transition_state(i,-1) == i # P-branch 
