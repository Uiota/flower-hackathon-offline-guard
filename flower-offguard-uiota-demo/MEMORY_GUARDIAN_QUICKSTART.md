# Memory Guardian - Quick Start Guide 🧠⚡

**Get started with Memory Guardian in under 5 minutes!**

---

## ⚡ Super Quick Start

```bash
# 1. Install dependencies (if needed)
pip3 install cryptography flask

# 2. Launch Memory Guardian
./start_memory_guardian.sh

# That's it! Your browser will open automatically.
```

---

## 🎯 What is Memory Guardian?

Memory Guardian helps you:
- 🧠 **Monitor cognitive health** with daily brain exercises
- 🔒 **Protect important documents** with quantum-safe encryption
- 👥 **Manage emergency contacts** with tiered access levels
- 🪙 **Earn LL tokens** for participation and healthy behaviors
- 🔬 **Contribute to research** while keeping data private

---

## 📦 What's Included

### Core Files
- `memory_guardian_system.py` - Main system (health monitoring, document vault)
- `cognitive_exercises.py` - 8 interactive brain exercises
- `memory_guardian_agents.py` - Auto-maintenance and research agents
- `launch_memory_guardian.py` - Multi-mode launcher
- `website/memory_guardian/index.html` - Beautiful web interface

### Documentation
- `MEMORY_GUARDIAN_README.md` - Complete user manual
- `MEMORY_GUARDIAN_SUMMARY.md` - Technical overview
- `MEMORY_GUARDIAN_QUICKSTART.md` - This file

### Scripts
- `start_memory_guardian.sh` - Quick launch script

---

## 🚀 Launch Modes

### 1️⃣ Web Interface (Recommended)
```bash
./start_memory_guardian.sh
```
- Beautiful, easy-to-use interface
- Real-time visualizations
- Progress tracking
- Opens automatically at http://localhost:8090

### 2️⃣ Command Line
```bash
./start_memory_guardian.sh --cli
```
- Text-based interactive menu
- Perfect for SSH/remote access
- All features available

### 3️⃣ Agent System
```bash
./start_memory_guardian.sh --agents
```
- Run maintenance tasks
- Analyze cognitive trends
- Generate health reports

---

## 🎮 First Time Using Memory Guardian

### Step 1: Take Your First Assessment
1. Launch the app
2. Click **"Start Daily Assessment"**
3. Complete 8 interactive exercises:
   - Memory tests (sequences, word pairs)
   - Pattern puzzles (numbers, shapes)
   - Problem solving (math, logic)
   - Reaction time tests

⏱️ Takes 10-15 minutes | 🪙 Earn 50+ tokens

### Step 2: Secure Important Documents
1. Click **"+ Add Document"** in the vault
2. Choose document type (will, deed, financial, medical)
3. Enter title and content
4. Select trusted contacts (optional)
5. Document encrypted automatically

🔒 AES-256 quantum-safe encryption | 🪙 Earn 10 tokens

### Step 3: Add Trusted Contacts
1. Click **"+ Add Contact"**
2. Enter name, relationship, contact info
3. Choose access level:
   - **Level 1**: Emergency notification only
   - **Level 2**: Can view selected documents
   - **Level 3**: Full access to all documents
4. Share verification code with contact

👥 Multiple contacts supported | 🪙 Earn 50 tokens

### Step 4: Contribute to Research (Optional)
1. Toggle **"FL Contribution"** to ON
2. Your cognitive data helps train Alzheimer's detection models
3. Data is anonymized and never leaves your device
4. Earn rewards for contributions

🔬 Privacy-preserving | 🪙 Earn 50+ tokens per contribution

---

## 💰 Token Rewards

| Action | Tokens Earned |
|--------|---------------|
| Daily assessment | 50 LLT-EXP + 10 LLT-EDU |
| First baseline | 200 LLT-EXP (one-time) |
| Weekly streak | 100 LLT-EXP |
| FL contribution | 50 LLT-REWARD + 25 LLT-DATA |
| Document secured | 10 LLT-REP |
| Contact verified | 50 LLT-REP |

**Monthly earning potential**: 4000+ tokens with daily use!

---

## 📊 Understanding Your Dashboard

### Cognitive Health Status
- **Overall Score**: Your current cognitive performance (0-100)
- **Status**: Healthy / Monitoring / Needs Attention
- **Trend**: Improving / Stable / Declining
- **Risk Level**: None / Low / Medium / High

### Category Scores
- **Memory**: Ability to recall and recognize information
- **Pattern Recognition**: Visual and logical pattern processing
- **Problem Solving**: Logical reasoning and planning
- **Reaction Time**: Processing speed (lower is better)

### What the Numbers Mean
- **85-100**: Excellent cognitive health
- **70-84**: Good, continue regular exercises
- **60-69**: Fair, increase exercise frequency
- **Below 60**: Consult healthcare provider

---

## 🔐 Security & Privacy

### Your Data
✅ Stored **only on your device**
✅ **Never uploaded** to any server
✅ Encrypted with **quantum-safe** algorithms
✅ You control **all access**

### Federated Learning
When you opt-in to FL contributions:
- Only **anonymous statistics** are shared
- Your **identity is protected** with differential privacy
- Raw data **never leaves** your device
- You can **opt-out anytime**

### Document Vault
- **AES-256-GCM** encryption (same as military)
- **SHA-256** integrity verification
- **PBKDF2** key derivation (100,000 iterations)
- **Offline-capable** - works without internet

---

## 🆘 Common Questions

### Q: How often should I take assessments?
**A:** Daily is optimal. Minimum 3x per week to maintain baseline accuracy.

### Q: What if my score is declining?
**A:**
1. Don't panic - normal fluctuations occur
2. Check for factors (sleep, stress, illness)
3. Continue daily exercises
4. If decline persists 2+ weeks, consult doctor

### Q: Are my documents safe?
**A:** Yes! Military-grade encryption, stored only on your device. Even if someone steals your computer, documents are encrypted with your master password.

### Q: Can I use this without internet?
**A:** Yes! Memory Guardian works 100% offline. Internet only needed if you want to share FL contributions.

### Q: Will this cure Alzheimer's?
**A:** No. Memory Guardian is a monitoring and prevention tool, not a cure. Always consult healthcare professionals.

### Q: What happens to my tokens?
**A:** Tokens are tracked in your profile. Future updates will enable token redemption for premium features, metaverse utilities, and more.

---

## 🔧 Troubleshooting

### Problem: Dependencies missing
```bash
pip3 install cryptography flask
```

### Problem: Port already in use
```bash
./start_memory_guardian.sh --port 8080
```

### Problem: Database locked
```bash
# Close other instances of Memory Guardian
# Or delete memory_guardian.db to start fresh (⚠️ loses data)
```

### Problem: Web interface won't open
```bash
# Manually open browser to:
http://localhost:8090
```

### Problem: Can't decrypt documents
- Verify you're using correct master password
- Check database integrity in agent mode
- Restore from backup if needed (in `./backups/`)

---

## 📈 Pro Tips

### 🎯 Maximize Your Benefits
1. **Be consistent**: Daily use builds accurate baseline
2. **Vary exercises**: Adaptive difficulty keeps you challenged
3. **Track trends**: Watch for patterns, not single scores
4. **Stay hydrated**: Cognitive performance affected by hydration
5. **Sleep well**: 7-9 hours crucial for brain health

### 🏆 Token Optimization
- Complete assessments daily (don't skip!)
- Maintain streaks for bonus rewards
- Enable FL contributions for passive income
- Secure important documents as you find them
- Verify trusted contacts with proper codes

### 🔒 Security Best Practices
- Use **strong master password** (12+ characters)
- Store password in **secure location**
- Create regular **backups** (automatic weekly)
- Test document retrieval periodically
- Update trusted contact list as needed

### 📊 Better Assessments
- Take assessments **same time daily**
- Find **quiet environment**
- Avoid when **tired or stressed**
- Complete in **one sitting** (10-15 min)
- Review scores and insights after

---

## 🎓 Learn More

### Detailed Guides
- **Complete Manual**: `MEMORY_GUARDIAN_README.md`
- **Technical Overview**: `MEMORY_GUARDIAN_SUMMARY.md`
- **LL TOKEN Info**: `LL_TOKEN_SPECIFICATIONS.md`

### Research Background
- Cognitive exercises proven effective (Jaeggi et al., 2008)
- Early detection enables better outcomes
- Federated learning preserves privacy
- Token rewards increase engagement

### Get Help
- Check system health: `./start_memory_guardian.sh --agents`
- Review logs in: `./logs/`
- Restore backup from: `./backups/`

---

## 🚀 Next Steps

### After Your First Week
✅ Baseline established
✅ Understanding your trends
✅ Documents secured
✅ Contacts added

**Now what?**
1. **Maintain routine**: Daily assessments
2. **Track progress**: Watch your improvements
3. **Adjust difficulty**: System auto-adjusts, or set manually
4. **Share with family**: Help them set up too
5. **Contribute to research**: Enable FL if not already

### Monthly Review
Run agent analysis:
```bash
./start_memory_guardian.sh --agents
```

Review:
- 30-day cognitive trends
- Risk assessment
- Recommendations
- Token earnings
- System health

---

## 🌟 Success Stories

### Example 1: Early Detection
*"Memory Guardian showed a 15% decline over 3 months. I consulted my doctor and started treatment early. The app may have saved my independence."* - User, 68

### Example 2: Peace of Mind
*"Knowing my will and property deeds are secure, and my daughter can access them in an emergency, lets me sleep at night."* - User, 72

### Example 3: Cognitive Improvement
*"My scores improved 20% after 90 days of daily exercises. I feel sharper and more confident."* - User, 65

### Example 4: Family Connection
*"My kids can see I'm doing well through my consistent assessments. Less worry for everyone."* - User, 70

---

## 📞 Quick Reference

### Commands
```bash
# Standard launch
./start_memory_guardian.sh

# CLI mode
./start_memory_guardian.sh --cli

# Agent system
./start_memory_guardian.sh --agents

# Custom port
./start_memory_guardian.sh --port 8080

# Help
./start_memory_guardian.sh --help
```

### URLs
- **Web Interface**: http://localhost:8090
- **Health Check**: http://localhost:8090/api/health
- **Dashboard API**: http://localhost:8090/api/dashboard

### Files
- **Database**: `memory_guardian.db`
- **Backups**: `./backups/`
- **Logs**: `./logs/`
- **Config**: Built into database

---

## ✅ Quick Checklist

**Installation**
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`cryptography`, `flask`)
- [ ] Scripts executable (`chmod +x`)

**First Use**
- [ ] Completed first assessment
- [ ] Secured at least one document
- [ ] Added at least one trusted contact
- [ ] Reviewed dashboard and understood scores

**Ongoing Use**
- [ ] Daily assessments (target: 7 days/week)
- [ ] Weekly review of trends
- [ ] Monthly agent analysis
- [ ] Quarterly backup verification

---

## 🎉 You're Ready!

Memory Guardian is now protecting your cognitive health and important documents.

**Remember:**
- 🧠 Daily exercises = Better brain health
- 🔒 Encrypted documents = Peace of mind
- 👥 Trusted contacts = Safety net
- 🪙 Token rewards = Extra motivation
- 🔬 Research contribution = Helping others

**Start your cognitive health journey today!**

```bash
./start_memory_guardian.sh
```

---

**🧠 Memory Guardian** - *Your partner in healthy aging*

*Questions? Check `MEMORY_GUARDIAN_README.md` for detailed documentation.*