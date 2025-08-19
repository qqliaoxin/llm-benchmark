import asyncio
import time
import json
import argparse
import random
import logging
from openai import AsyncOpenAI
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# 尝试导入tiktoken进行更精确的token计算
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.info("tiktoken not available, using character-based estimation")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_tokens_accurate(text, model_name="gpt-3.5-turbo"):
    """更精确的token计算函数"""
    if not text:
        return 0
        
    if TIKTOKEN_AVAILABLE:
        try:
            # 尝试使用tiktoken进行精确计算
            if "qwen" in model_name.lower():
                # Qwen模型使用cl100k_base编码
                encoding = tiktoken.get_encoding("cl100k_base")
            elif "deepseek" in model_name.lower():
                # DeepSeek模型也使用cl100k_base编码
                encoding = tiktoken.get_encoding("cl100k_base")
            else:
                # 默认使用cl100k_base编码
                encoding = tiktoken.get_encoding("cl100k_base")
            
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logging.debug(f"tiktoken encoding failed: {e}, falling back to estimation")
    
    # 回退到改进的字符估算方法
    chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    english_chars = len(text) - chinese_chars
    
    # 更精确的token估算
    chinese_tokens = chinese_chars / 1.5 if chinese_chars > 0 else 0
    english_tokens = english_chars / 4.0 if english_chars > 0 else 0
    
    return max(1, int(chinese_tokens + english_tokens))

# 不同大小的上下文模板
CONTEXT_TEMPLATES = {
    "13t": {
        "size": "13t",
        "context": "请重复这句话：这是一个测试"
    },
    "1k": {
        "size": "1k",
        "context": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的智能机器。这些任务包括学习、推理、问题解决、感知和语言理解。AI系统可以分为两大类：窄AI（专门设计用于特定任务）和通用AI（具有类似人类的认知能力）。机器学习是AI的一个子集，使计算机能够从数据中学习而无需明确编程。深度学习进一步利用神经网络来模拟人脑的工作方式。AI在各个领域都有应用，包括医疗保健、金融、交通和娱乐。随着技术的进步，AI正在改变我们的工作和生活方式，但也带来了关于就业、隐私和伦理的重要问题。" * 2
    },
    "2k": {
        "size": "2k", 
        "context": "气候变化是指地球气候系统长期的变化，主要由人类活动引起的温室气体排放所驱动。自工业革命以来，大气中二氧化碳的浓度急剧增加，主要来源于化石燃料的燃烧、森林砍伐和工业过程。这导致了全球平均温度的上升，被称为全球变暖。气候变化的影响是多方面的：海平面上升威胁沿海地区，极端天气事件变得更加频繁和严重，生态系统受到破坏，农业生产面临挑战。北极冰盖融化，永久冻土层解冻，这些都进一步加剧了气候变化。为了应对这一挑战，国际社会制定了《巴黎协定》等国际协议，旨在限制全球温升。减缓气候变化需要转向可再生能源，提高能源效率，发展碳捕获技术，以及改变生活方式。适应气候变化同样重要，包括建设抗灾基础设施，开发抗旱作物，以及制定应急预案。每个人都可以通过减少碳足迹来为应对气候变化做出贡献。" * 3
    },
    "4k": {
        "size": "4k",
        "context": "量子计算是一种利用量子力学现象进行信息处理的计算范式。与传统计算机使用比特（0或1）不同，量子计算机使用量子比特（qubits），它们可以同时处于0和1的叠加状态。这种特性，加上量子纠缠和量子干涉，使量子计算机能够并行处理大量信息，在某些问题上提供指数级的速度提升。量子计算的核心概念包括叠加态、纠缠和测量。叠加态允许量子比特同时存在于多个状态，而纠缠则创建了量子比特之间的强相关性，即使它们在物理上分离。量子算法如Shor算法（用于因数分解）和Grover算法（用于搜索）展示了量子计算的潜力。然而，量子计算面临重大挑战，包括量子退相干、错误率高和需要极低温度的操作环境。目前的量子计算机仍处于噪声中等规模量子（NISQ）时代，容易受到环境干扰。尽管如此，IBM、Google、Microsoft等公司正在积极开发量子计算技术。量子计算的潜在应用包括密码学、药物发现、金融建模、人工智能和材料科学。量子密码学已经在某些领域得到应用，提供了理论上不可破解的通信安全。随着技术的进步，量子计算有望在未来几十年内解决一些最复杂的计算问题，但也可能对现有的加密方法构成威胁。" * 4
    },
    "8k": {
        "size": "8k",
        "context": "生物技术是一个跨学科领域，结合了生物学、化学、物理学、工程学和计算机科学，利用生物系统、生物体或其衍生物来开发或制造产品。现代生物技术的发展可以追溯到DNA重组技术的发明，这使得科学家能够操纵基因并创造转基因生物。基因工程技术包括PCR（聚合酶链反应）、基因克隆、基因测序和CRISPR-Cas9基因编辑系统。CRISPR技术特别革命性，因为它允许精确、高效且相对便宜的基因编辑。生物技术在医学领域有广泛应用，包括重组蛋白药物的生产、基因治疗、细胞治疗和个性化医学。胰岛素、生长激素和单克隆抗体等药物都是通过生物技术生产的。在农业方面，生物技术用于开发抗虫、抗除草剂和营养强化的转基因作物。工业生物技术利用微生物和酶来生产化学品、燃料和材料，提供了更环保的替代方案。合成生物学是生物技术的新兴分支，旨在设计和构建新的生物部件、设备和系统。环境生物技术用于污染治理、废物处理和环境监测。生物信息学结合了生物学和计算机科学，用于分析和解释生物数据。然而，生物技术也引发了伦理、安全和监管方面的担忧，包括转基因食品的安全性、基因编辑的伦理界限以及生物武器的潜在风险。随着技术的不断进步，生物技术有望在解决全球健康、食品安全和环境挑战方面发挥越来越重要的作用。" * 6
    },
    "16k": {
        "size": "16k",
        "context": "区块链技术是一种分布式账本技术，通过密码学方法将数据块按时间顺序链接起来，形成一个不可篡改的数据链。每个区块包含前一个区块的哈希值、时间戳和交易数据，这种设计确保了数据的完整性和不可篡改性。区块链的核心特征包括去中心化、透明性、不可篡改性和共识机制。去中心化意味着没有单一的控制点，网络由多个节点共同维护。透明性确保所有交易都是公开可验证的，而共识机制（如工作量证明、权益证明）确保网络参与者对账本状态达成一致。比特币是第一个成功的区块链应用，展示了数字货币的可能性。以太坊进一步扩展了区块链的功能，引入了智能合约概念，允许在区块链上执行自动化的合约逻辑。智能合约是自执行的合约，其条款直接写入代码中，当预定条件满足时自动执行。这开启了去中心化应用（DApps）的时代，包括去中心化金融（DeFi）、非同质化代币（NFT）和去中心化自治组织（DAO）。DeFi重新构想了传统金融服务，提供去中心化的借贷、交易和保险服务。NFT为数字艺术品和收藏品创造了新的市场。区块链技术在供应链管理中也有重要应用，提供产品从生产到消费的完整可追溯性。在医疗保健领域，区块链可以安全地存储和共享患者数据。数字身份管理是另一个重要应用，用户可以控制自己的身份信息而不依赖中心化机构。然而，区块链技术也面临挑战，包括可扩展性问题、能源消耗、监管不确定性和用户体验问题。第二层解决方案如闪电网络和侧链正在解决可扩展性问题。权益证明等新的共识机制正在减少能源消耗。随着技术的成熟和监管框架的完善，区块链有望在更多领域得到应用，推动数字经济的发展。" * 8
    },
    "32k": {
        "size": "32k",
        "context": "神经科学是研究神经系统结构和功能的科学领域，涵盖了从分子和细胞水平到行为和认知水平的各个层面。人类大脑包含约860亿个神经元，通过数万亿个突触连接形成复杂的神经网络。神经元是神经系统的基本单位，通过电化学信号进行通信。神经元的结构包括细胞体、树突和轴突，其中树突接收信号，轴突传递信号。突触是神经元之间的连接点，通过神经递质进行化学传递。大脑的结构高度复杂，包括大脑皮层、小脑、脑干和边缘系统等主要区域。大脑皮层负责高级认知功能，如思维、语言和意识。小脑主要负责运动协调和平衡。脑干控制基本生命功能，如呼吸和心跳。边缘系统参与情绪、记忆和动机。神经可塑性是大脑的一个重要特性，指神经系统根据经验改变其结构和功能的能力。这种可塑性是学习和记忆的基础，也是大脑损伤后康复的机制。记忆形成涉及多个大脑区域的协调工作，包括海马体、杏仁核和新皮层。长期记忆的形成需要蛋白质合成和突触强度的持久改变。神经科学研究方法包括电生理学、神经影像学、光遗传学和分子生物学技术。功能性磁共振成像（fMRI）和正电子发射断层扫描（PET）等技术使我们能够观察活体大脑的活动。光遗传学技术允许科学家用光精确控制特定神经元的活动。神经疾病如阿尔茨海默病、帕金森病、抑郁症和精神分裂症严重影响人类健康。这些疾病的研究推动了我们对大脑功能的理解，也促进了新治疗方法的开发。深度脑刺激、药物治疗和认知行为疗法等治疗手段正在不断改进。计算神经科学结合数学模型和计算机模拟来理解大脑功能，这一领域也推动了人工智能的发展。神经网络算法受到生物神经网络的启发，在机器学习中取得了巨大成功。脑机接口技术正在开发中，有望帮助瘫痪患者恢复运动能力。意识研究是神经科学的前沿领域，试图理解主观体验的神经基础。这涉及到哲学、心理学和神经科学的交叉。随着技术的进步，我们对大脑的理解不断深入，这不仅有助于治疗神经疾病，也可能揭示人类认知和意识的奥秘。" * 12
    },
    "64k": {
        "size": "64k",
        "context": "宇宙学是研究宇宙整体结构、起源、演化和最终命运的科学。现代宇宙学建立在爱因斯坦的广义相对论基础上，结合了天体物理学、粒子物理学和观测天文学的最新发现。大爆炸理论是目前最被广泛接受的宇宙起源模型，认为宇宙始于约138亿年前的一个极其炽热和致密的状态，然后经历了快速膨胀和冷却过程。宇宙微波背景辐射的发现为大爆炸理论提供了强有力的证据，这是宇宙早期留下的余辉。暗物质和暗能量是现代宇宙学中最神秘的组成部分，它们分别占宇宙总质量能量的约27%和68%，而我们熟悉的普通物质仅占约5%。暗物质通过引力效应影响星系的形成和演化，但不与电磁辐射相互作用，因此无法直接观测。暗能量被认为是导致宇宙加速膨胀的原因，但其本质仍然是个谜。宇宙的大尺度结构呈现出网状分布，由星系团、星系群和巨大的空洞组成。这种结构的形成可以追溯到宇宙早期的微小密度涨落，这些涨落在引力作用下逐渐放大，最终形成了我们今天观察到的宇宙结构。恒星的生命周期对宇宙的化学演化起着关键作用，重元素在恒星内部通过核聚变产生，并在超新星爆发时散布到宇宙中，为后续恒星和行星的形成提供了原料。黑洞是宇宙中最极端的天体，具有如此强大的引力场，连光都无法逃脱。超大质量黑洞位于大多数星系的中心，对星系的演化产生重要影响。引力波的探测开启了观测宇宙的新窗口，使我们能够研究黑洞合并、中子星碰撞等极端事件。宇宙学常数问题、层次问题和暗物质的本质等基本问题仍然困扰着物理学家。多元宇宙理论提出我们的宇宙可能只是无数宇宙中的一个，但这一理论目前还无法通过实验验证。量子引力理论试图统一广义相对论和量子力学，可能为理解宇宙的最初时刻提供新的洞察。宇宙的最终命运取决于暗能量的性质，可能的结局包括热寂、大撕裂或大坍缩。随着观测技术的不断进步，如詹姆斯·韦伯太空望远镜等新一代设备，我们对宇宙的理解将继续深化。" * 16
    },
    "92k": {
        "size": "92k",
        "context": "进化生物学是研究生物多样性起源和发展的科学领域，探索生命如何从简单的形式演化为今天我们看到的复杂多样的生物世界。达尔文的自然选择理论为现代进化生物学奠定了基础，该理论提出具有有利变异的个体更可能生存和繁殖，从而将这些特征传递给后代。现代综合理论将达尔文的自然选择与孟德尔遗传学、分子生物学和群体遗传学相结合，形成了更完整的进化框架。DNA和RNA的发现揭示了遗传信息的分子基础，使我们能够在基因水平上理解进化过程。分子钟技术通过比较不同物种间的基因序列差异来估算它们的分化时间，为构建生命树提供了重要工具。化石记录虽然不完整，但为我们提供了生命演化历史的直接证据，展示了从简单的单细胞生物到复杂多细胞生物的演化过程。寒武纪大爆发是生命史上一个重要事件，在相对较短的地质时间内出现了大量新的动物门类。大灭绝事件如二叠纪末大灭绝和白垩纪末大灭绝深刻影响了生命的演化轨迹，为新的生物类群的辐射演化创造了机会。适应性辐射是指一个祖先物种在短时间内分化为多个适应不同生态位的后代物种，加拉帕戈斯雀鸟是经典的例子。共同演化描述了不同物种之间相互影响的演化过程，如植物与传粉者、捕食者与猎物之间的协同演化。性选择是自然选择的一种特殊形式，解释了许多看似不利于生存的特征的演化，如孔雀的华丽尾羽。基因漂变在小群体中起重要作用，可能导致有害基因的固定或有利基因的丢失。分子进化研究揭示了基因和蛋白质的演化模式，发现了中性演化和正选择的证据。比较基因组学通过比较不同物种的基因组序列，揭示了基因功能的演化和新基因的起源。表观遗传学发现环境因素可以影响基因表达而不改变DNA序列，这些变化有时可以遗传给后代，为演化提供了新的机制。发育生物学与进化生物学的结合产生了进化发育生物学（evo-devo），研究发育过程的演化如何产生形态多样性。人类演化是进化生物学的一个重要分支，通过化石证据、基因分析和比较解剖学研究人类的起源和演化历程。现代人类的非洲起源理论得到了遗传学证据的强力支持，显示所有现代人类都起源于约20万年前的非洲。文化演化研究人类文化特征的传播和变化，发现文化演化遵循类似生物演化的规律。保护生物学应用进化原理来保护濒危物种和生态系统，强调遗传多样性对物种长期生存的重要性。气候变化对当代演化产生重要影响，许多物种正在经历快速的适应性演化以应对环境变化。" * 20
    },
    "128k": {
        "size": "128k",
        "context": "系统生物学是一个跨学科的研究领域，旨在通过整合分子、细胞、组织和器官水平的信息来理解生物系统的复杂性和功能。这个领域的出现源于认识到生物系统的特性不能仅通过研究其组成部分来理解，而需要考虑这些部分之间的相互作用和系统的整体行为。系统生物学采用定量和计算方法来建模和分析生物网络，包括基因调控网络、蛋白质相互作用网络、代谢网络和信号传导网络。高通量技术如基因组学、转录组学、蛋白质组学和代谢组学为系统生物学提供了大量数据，使研究人员能够同时监测数千个分子的活动。生物信息学和计算生物学是系统生物学的核心工具，用于处理和分析这些大规模数据集。网络生物学研究生物分子之间的相互作用网络，发现了许多重要的网络特性，如小世界特性、无标度分布和模块化结构。这些网络特性反映了生物系统的鲁棒性和效率。基因调控网络控制基因的表达模式，决定细胞的身份和功能。转录因子、microRNA和表观遗传修饰都参与基因调控的复杂网络。信号传导网络使细胞能够感知环境变化并做出适当反应，这些网络通常具有多层次的调控机制和反馈回路。代谢网络描述了细胞内化学反应的相互关系，代谢流分析可以预测细胞在不同条件下的代谢状态。系统生物学在疾病研究中发挥重要作用，通过分析疾病相关的分子网络变化来理解疾病机制。癌症系统生物学研究肿瘤发生发展过程中的网络重构，为开发新的治疗策略提供指导。药物系统生物学研究药物对生物网络的影响，有助于药物发现和个性化医疗的发展。合成生物学是系统生物学的应用分支，旨在设计和构建具有特定功能的生物系统。这个领域结合了工程学原理和生物学知识，开发标准化的生物部件和模块。定量生物学强调使用数学模型和定量测量来理解生物过程，这种方法有助于发现生物系统中的定量规律和原理。单细胞技术的发展使研究人员能够在单个细胞水平上研究生物系统，揭示了细胞间的异质性和动态变化。时间序列分析用于研究生物系统的动态行为，如细胞周期、昼夜节律和发育过程。多尺度建模试图连接不同生物组织层次的模型，从分子到细胞再到组织和器官。机器学习和人工智能在系统生物学中的应用越来越广泛，用于模式识别、预测建模和数据挖掘。个性化医疗是系统生物学的重要应用目标，通过分析个体的分子特征来制定个性化的治疗方案。系统免疫学研究免疫系统的网络特性，有助于理解免疫反应的调控机制和开发新的免疫疗法。农业系统生物学应用系统方法来改良作物，提高产量和抗性。环境系统生物学研究生物系统对环境变化的响应，为环境保护和可持续发展提供科学依据。随着技术的不断进步和数据的积累，系统生物学有望为理解生命的复杂性和解决人类面临的重大挑战做出更大贡献。" * 24
    }
}

# 测试问题模板
TEST_QUESTIONS = [
    "请根据上述内容，总结主要观点。",
    "基于提供的信息，分析其中的关键概念。",
    "请解释上述内容中最重要的三个要点。",
    "根据上下文，这个领域面临的主要挑战是什么？",
    "请简要概括上述内容的核心思想。",
    "基于提供的信息，这个主题的未来发展趋势如何？",
    "请分析上述内容中提到的技术或概念的优缺点。",
    "根据上下文，这个领域对社会的影响是什么？"
]

async def process_stream(stream, model_name="gpt-3.5-turbo"):
    """处理流式响应并计算指标"""
    first_token_time = None
    total_content = ""
    total_reasoning = ""
    chunk_count = 0
    
    try:
        async for chunk in stream:
            chunk_count += 1
            
            if first_token_time is None:
                first_token_time = time.time()
            
            # 收集内容
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                total_content += chunk.choices[0].delta.content
            
            if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[0].delta.reasoning_content:
                total_reasoning += chunk.choices[0].delta.reasoning_content
            
            if chunk.choices[0].finish_reason is not None:
                break
        
        # 使用改进的token计算方法
        content_tokens = calculate_tokens_accurate(total_content, model_name) if total_content else 0
        reasoning_tokens = calculate_tokens_accurate(total_reasoning, model_name) if total_reasoning else 0
            
        total_tokens = content_tokens + reasoning_tokens
        
        # 如果没有收到任何内容，但收到了 chunk，说明可能有问题
        if chunk_count > 0 and total_tokens == 0:
            # 至少返回 1 个 token，表示收到了响应
            total_tokens = 1
            content_tokens = 1
        
        # 详细的调试信息
        if total_content:
            chinese_chars = len([c for c in total_content if '\u4e00' <= c <= '\u9fff'])
            english_chars = len(total_content) - chinese_chars
            logging.debug(f"Content analysis: total_len={len(total_content)}, chinese_chars={chinese_chars}, english_chars={english_chars}")
        
        logging.debug(f"Stream processed: {chunk_count} chunks, content_len={len(total_content)}, reasoning_len={len(total_reasoning)}, content_tokens={content_tokens}, reasoning_tokens={reasoning_tokens}, total_tokens={total_tokens}")
        logging.debug(f"Content preview: {total_content[:100]}{'...' if len(total_content) > 100 else ''}")
        
        return first_token_time, total_tokens, content_tokens, reasoning_tokens
        
    except Exception as e:
        logging.error(f"Error processing stream: {e}")
        # 如果处理流时出错，但已经收到了第一个 token，仍然返回部分结果
        if first_token_time is not None:
            estimated_tokens = calculate_tokens_accurate(total_content, model_name) if total_content else 1
            return first_token_time, estimated_tokens, estimated_tokens, 0
        else:
            raise e

async def make_context_request(client, model, context_size, output_tokens, request_timeout, request_id=None):
    """发送带有指定上下文大小的请求"""
    start_time = time.time()
    
    # 获取对应大小的上下文
    context_template = CONTEXT_TEMPLATES[context_size]
    context_content = context_template["context"]
    
    # 对于 13t，不拼接额外的问题
    if context_size == "13t":
        full_prompt = context_content
        question = "内置问题（13t测试）"
    else:
        question = random.choice(TEST_QUESTIONS)
        # 组合完整的提示
        full_prompt = f"{context_content}\n\n{question}"
    
    # 计算上下文大小（字符数和token估算）
    context_char_count = len(context_content)
    prompt_char_count = len(full_prompt)
    # 使用改进的token计算方法
    prompt_tokens_estimate = calculate_tokens_accurate(full_prompt, model)
    
    try:
        logging.debug(f"Request {request_id}: Sending request with prompt length {len(full_prompt)}")
        
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=output_tokens,
            stream=True,
            # 添加 SSE 相关参数
            stream_options={"include_usage": True} if hasattr(client, 'stream_options') else None
        )
        
        logging.debug(f"Request {request_id}: Stream created, processing...")
        
        first_token_time, total_tokens, content_tokens, reasoning_tokens = await asyncio.wait_for(
            process_stream(stream, model), timeout=request_timeout
        )
        
        logging.debug(f"Request {request_id}: Stream processed successfully, tokens={total_tokens}")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else None
        
        # 计算不同的吞吐量指标
        generation_throughput = total_tokens / elapsed_time if elapsed_time > 0 and total_tokens > 0 else 0
        prompt_throughput = prompt_tokens_estimate / ttft if ttft and ttft > 0 else 0
        
        return {
            "success": True,
            "request_id": request_id,
            "context_size": context_size,
            "context_char_count": context_char_count,
            "prompt_char_count": prompt_char_count,
            "prompt_tokens_estimate": prompt_tokens_estimate,
            "total_tokens": total_tokens,
            "content_tokens": content_tokens,
            "reasoning_tokens": reasoning_tokens,
            "elapsed_time": elapsed_time,
            "generation_throughput": generation_throughput,
            "prompt_throughput": prompt_throughput,
            "ttft": ttft,
            "question": question,
            "start_time": start_time,
            "end_time": end_time
        }
        
    except asyncio.TimeoutError:
        logging.warning(f"Request {request_id} with context size {context_size} timed out after {request_timeout} seconds")
        return {
            "success": False,
            "request_id": request_id,
            "context_size": context_size,
            "error": "timeout",
            "context_char_count": context_char_count,
            "prompt_char_count": prompt_char_count,
            "prompt_tokens_estimate": prompt_tokens_estimate
        }
    except Exception as e:
        logging.error(f"Error during request {request_id} with context size {context_size}: {str(e)}")
        logging.error(f"Request details - prompt_length: {len(full_prompt)}, model: {model}")
        return {
            "success": False,
            "request_id": request_id,
            "context_size": context_size,
            "error": str(e),
            "error_type": type(e).__name__,
            "context_char_count": context_char_count,
            "prompt_char_count": prompt_char_count,
            "prompt_tokens_estimate": prompt_tokens_estimate
        }

async def test_sse_connection(llm_url, api_key, model):
    """测试 SSE 连接是否正常工作"""
    client = AsyncOpenAI(base_url=llm_url, api_key=api_key)
    
    try:
        print("\n🔍 测试 SSE 连接...")
        
        # 简单的测试请求
        test_prompt = "请说'你好'"
        
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": test_prompt}],
            max_tokens=10,
            stream=True
        )
        
        chunk_count = 0
        content_received = ""
        
        async for chunk in stream:
            chunk_count += 1
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                content_received += chunk.choices[0].delta.content
            
            if chunk.choices[0].finish_reason is not None:
                break
        
        if chunk_count > 0:
            print(f"✅ SSE 连接正常 - 收到 {chunk_count} 个数据块，内容长度: {len(content_received)}")
            if content_received:
                print(f"   收到内容: {content_received[:50]}{'...' if len(content_received) > 50 else ''}")
            return True
        else:
            print("❌ SSE 连接异常 - 未收到任何数据块")
            return False
            
    except Exception as e:
        print(f"❌ SSE 连接测试失败: {e}")
        return False

async def run_context_benchmark(context_sizes, num_requests_per_size, output_tokens, llm_url, api_key, model, request_timeout, concurrency=1):
    """运行上下文基准测试"""
    client = AsyncOpenAI(base_url=llm_url, api_key=api_key)
    all_results = []
    
    for context_size in context_sizes:
        print(f"\n测试上下文大小: {context_size} ({CONTEXT_TEMPLATES[context_size]['size']})")
        print(f"上下文字符数: {len(CONTEXT_TEMPLATES[context_size]['context']):,}")
        print(f"并发数: {concurrency}")
        
        size_results = []
        test_failed = False  # 标记当前测试是否失败
        
        # 如果并发数为1，按顺序执行
        if concurrency == 1:
            for i in range(num_requests_per_size):
                print(f"  请求 {i+1}/{num_requests_per_size}...", end=" ")
                result = await make_context_request(client, model, context_size, output_tokens, request_timeout, f"{context_size}-{i+1}")
                size_results.append(result)
                
                if result["success"]:
                    print(f"成功 - 延迟: {result['elapsed_time']:.2f}s, 生成TPS: {result['generation_throughput']:.1f}, TTFT: {result['ttft']:.3f}s")
                else:
                    print(f"失败 - {result.get('error', 'unknown error')}")
                    print(f"  检测到请求失败，停止当前上下文大小的后续请求")
                    test_failed = True
                    break  # 失败时停止发送后续请求
                
                # 请求间隔，避免过载
                await asyncio.sleep(1)
        else:
            # 并发执行
            print(f"  并发执行 {num_requests_per_size} 个请求...")
            
            # 创建并发任务
            tasks = []
            for i in range(num_requests_per_size):
                task = make_context_request(client, model, context_size, output_tokens, request_timeout, f"{context_size}-{i+1}")
                tasks.append(task)
            
            # 控制并发数
            semaphore = asyncio.Semaphore(concurrency)
            
            async def limited_request(task):
                async with semaphore:
                    return await task
            
            # 执行所有任务
            batch_start_time = time.time()
            results = await asyncio.gather(*[limited_request(task) for task in tasks], return_exceptions=True)
            batch_end_time = time.time()
            
            # 处理结果并检查失败
            has_failure = False
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    result = {
                        "success": False,
                        "request_id": f"{context_size}-{i+1}",
                        "context_size": context_size,
                        "error": str(result),
                        "context_char_count": len(CONTEXT_TEMPLATES[context_size]['context']),
                        "prompt_char_count": 0,
                        "prompt_tokens_estimate": 0
                    }
                    has_failure = True
                elif not result["success"]:
                    has_failure = True
                size_results.append(result)
            
            # 如果有失败，提示用户
            if has_failure:
                failed_count = len([r for r in size_results if not r["success"]])
                print(f"  检测到 {failed_count} 个请求失败，当前上下文大小测试完成")
                test_failed = True
            
            # 显示批次结果
            successful_requests = [r for r in size_results if r["success"]]
            batch_duration = batch_end_time - batch_start_time
            
            print(f"  批次完成 - 总时间: {batch_duration:.2f}s, 成功: {len(successful_requests)}/{len(size_results)}")
            if successful_requests:
                avg_ttft = np.mean([r['ttft'] for r in successful_requests if r['ttft'] is not None])
                avg_gen_tps = np.mean([r['generation_throughput'] for r in successful_requests])
                avg_prompt_tps = np.mean([r['prompt_throughput'] for r in successful_requests if r['prompt_throughput'] > 0])
                print(f"  平均指标 - TTFT: {avg_ttft:.3f}s, 生成TPS: {avg_gen_tps:.1f}, 提示TPS: {avg_prompt_tps:.1f}")
        
        all_results.append({
            "context_size": context_size,
            "concurrency": concurrency,
            "results": size_results
        })
        
        # 如果当前测试失败，停止测试后续的上下文大小
        if test_failed:
            print(f"\n检测到测试失败，停止测试后续的上下文大小")
            print(f"已完成的测试: {[result['context_size'] for result in all_results]}")
            break
    
    return all_results

def analyze_context_results(all_results):
    """分析上下文测试结果"""
    summary = []
    
    for size_group in all_results:
        context_size = size_group["context_size"]
        concurrency = size_group.get("concurrency", 1)
        results = size_group["results"]
        
        # 过滤成功的请求
        successful_results = [r for r in results if r["success"]]
        
        if not successful_results:
            summary.append({
                "context_size": context_size,
                "concurrency": concurrency,
                "success_rate": 0,
                "avg_latency": None,
                "avg_generation_tps": None,
                "avg_prompt_tps": None,
                "avg_ttft": None,
                "min_ttft": None,
                "max_ttft": None,
                "context_chars": results[0]["context_char_count"] if results else 0,
                "prompt_chars": results[0]["prompt_char_count"] if results else 0,
                "prompt_tokens": results[0]["prompt_tokens_estimate"] if results else 0,
                "total_requests": len(results),
                "successful_requests": 0
            })
            continue
        
        # 计算统计指标
        success_rate = len(successful_results) / len(results) * 100
        latencies = [r["elapsed_time"] for r in successful_results]
        generation_tps_values = [r["generation_throughput"] for r in successful_results]
        prompt_tps_values = [r["prompt_throughput"] for r in successful_results if r["prompt_throughput"] > 0]
        ttft_values = [r["ttft"] for r in successful_results if r["ttft"] is not None]
        
        avg_latency = np.mean(latencies) if latencies else None
        avg_generation_tps = np.mean(generation_tps_values) if generation_tps_values else None
        avg_prompt_tps = np.mean(prompt_tps_values) if prompt_tps_values else None
        avg_ttft = np.mean(ttft_values) if ttft_values else None
        
        p95_latency = np.percentile(latencies, 95) if latencies else None
        p95_generation_tps = np.percentile(generation_tps_values, 95) if generation_tps_values else None
        p95_prompt_tps = np.percentile(prompt_tps_values, 95) if prompt_tps_values else None
        p95_ttft = np.percentile(ttft_values, 95) if ttft_values else None
        
        # 计算最小和最大TTFT
        min_ttft = np.min(ttft_values) if ttft_values else None
        max_ttft = np.max(ttft_values) if ttft_values else None
        
        summary.append({
            "context_size": context_size,
            "concurrency": concurrency,
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "p95_latency": p95_latency,
            "avg_generation_tps": avg_generation_tps,
            "p95_generation_tps": p95_generation_tps,
            "avg_prompt_tps": avg_prompt_tps,
            "p95_prompt_tps": p95_prompt_tps,
            "avg_ttft": avg_ttft,
            "p95_ttft": p95_ttft,
            "min_ttft": min_ttft,
            "max_ttft": max_ttft,
            "context_chars": successful_results[0]["context_char_count"],
            "prompt_chars": successful_results[0]["prompt_char_count"],
            "prompt_tokens": successful_results[0]["prompt_tokens_estimate"],
            "total_requests": len(results),
            "successful_requests": len(successful_results)
        })
    
    return summary

def print_context_summary(summary, model_name):
    """打印上下文测试结果汇总"""
    console = Console(width=120)
    
    # 创建标题面板
    title = Text("LLM 上下文大小性能测试报告", style="bold")
    console.print(Panel(title, width=80))
    
    # 打印基本信息
    basic_info = Table(show_header=False, width=60)
    basic_info.add_column("项目", style="cyan", width=20)
    basic_info.add_column("值", style="green", width=40)
    
    basic_info.add_row("测试模型", model_name)
    basic_info.add_row("测试类型", "上下文大小性能测试")
    basic_info.add_row("测试时间", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    console.print("\n基本信息:")
    console.print(basic_info)
    
    # 创建详细结果表格
    table = Table(
        title="上下文大小性能对比",
        show_header=True,
        header_style="bold cyan",
        border_style="blue",
        width=120
    )
    
    # 添加列
    table.add_column("上下文大小", justify="center", style="cyan", width=8)
    table.add_column("并发数", justify="center", width=6)
    table.add_column("字符数", justify="right", width=8)
    table.add_column("成功率", justify="right", width=6)
    table.add_column("平均延迟(s)", justify="right", width=10)
    table.add_column("生成TPS", justify="right", width=8)
    table.add_column("提示TPS", justify="right", width=8)
    table.add_column("最小TTFT(s)", justify="right", width=10)
    table.add_column("最大TTFT(s)", justify="right", width=10)
    table.add_column("平均TTFT(s)", justify="right", width=10)
    
    # 添加数据行
    for row in summary:
        # 根据成功率设置行样式
        success_rate = row["success_rate"]
        row_style = "green" if success_rate >= 95 else "yellow" if success_rate >= 80 else "red"
        
        table.add_row(
            row["context_size"],
            str(row["concurrency"]),
            f"{row['context_chars']:,}",
            f"{success_rate:.1f}%",
            f"{row['avg_latency']:.3f}" if row['avg_latency'] is not None else "N/A",
            f"{row['avg_generation_tps']:.1f}" if row['avg_generation_tps'] is not None else "N/A",
            f"{row['avg_prompt_tps']:.1f}" if row['avg_prompt_tps'] is not None else "N/A",
            f"{row['min_ttft']:.3f}" if row['min_ttft'] is not None else "N/A",
            f"{row['max_ttft']:.3f}" if row['max_ttft'] is not None else "N/A",
            f"{row['avg_ttft']:.3f}" if row['avg_ttft'] is not None else "N/A",
            style=row_style
        )
    
    console.print("\n")
    console.print(table)
    
    # 性能分析
    valid_results = [r for r in summary if r["avg_latency"] is not None]
    if valid_results:
        console.print("\n性能分析:", style="bold cyan")
        
        # 找出最佳和最差性能
        best_latency = min(valid_results, key=lambda x: x["avg_latency"])
        worst_latency = max(valid_results, key=lambda x: x["avg_latency"])
        best_generation_tps = max(valid_results, key=lambda x: x["avg_generation_tps"] or 0)
        best_ttft = min([r for r in valid_results if r["avg_ttft"] is not None], key=lambda x: x["avg_ttft"], default=None)
        
        console.print(f"• 最低延迟: {best_latency['context_size']} ({best_latency['avg_latency']:.3f}s)", style="green")
        console.print(f"• 最高延迟: {worst_latency['context_size']} ({worst_latency['avg_latency']:.3f}s)", style="red")
        console.print(f"• 最高生成TPS: {best_generation_tps['context_size']} ({best_generation_tps['avg_generation_tps']:.1f} tokens/s)", style="green")
        if best_ttft:
            console.print(f"• 最佳TTFT: {best_ttft['context_size']} ({best_ttft['avg_ttft']:.3f}s)", style="green")
        
        # 延迟增长分析
        if len(valid_results) > 1:
            latency_increase = (worst_latency["avg_latency"] - best_latency["avg_latency"]) / best_latency["avg_latency"] * 100
            console.print(f"• 延迟增长: {latency_increase:.1f}% (从最小到最大上下文)", style="yellow")

def main():
    parser = argparse.ArgumentParser(description="测试 LLM 模型在不同上下文大小下的性能")
    parser.add_argument("--llm_url", type=str, required=True, help="LLM 服务器 URL")
    parser.add_argument("--api_key", type=str, required=False, default="default", help="API 密钥")
    parser.add_argument("--model", type=str, default="deepseek-r1", help="模型名称 (默认: deepseek-r1)")
    parser.add_argument("--context_sizes", type=str, default="13t,1k,2k,4k,8k,16k,32k,64k,92k,128k", 
                       help="要测试的上下文大小，用逗号分隔 (默认: 13t,1k,2k,4k,8k,16k,32k,64k,92k,128k)")
    parser.add_argument("--num_requests", type=int, default=3, 
                       help="每个上下文大小的请求次数 (默认: 3)")
    parser.add_argument("--output_tokens", type=int, default=200, 
                       help="输出 token 数量 (默认: 200)")
    parser.add_argument("--request_timeout", type=int, default=120, 
                       help="请求超时时间（秒） (默认: 120)")
    parser.add_argument("--concurrency", type=int, default=1, 
                       help="并发请求数 (默认: 1)")
    parser.add_argument("--debug", action="store_true", 
                       help="启用调试模式，显示详细日志")
    parser.add_argument("--skip_sse_test", action="store_true", 
                       help="跳过 SSE 连接测试")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 解析上下文大小
    context_sizes = [size.strip() for size in args.context_sizes.split(",")]
    
    # 验证上下文大小
    valid_sizes = list(CONTEXT_TEMPLATES.keys())
    for size in context_sizes:
        if size not in valid_sizes:
            print(f"错误: 无效的上下文大小 '{size}'. 可用选项: {', '.join(valid_sizes)}")
            return
    
    print(f"开始上下文性能测试...")
    print(f"模型: {args.model}")
    print(f"测试的上下文大小: {', '.join(context_sizes)}")
    print(f"每个大小的请求次数: {args.num_requests}")
    print(f"输出 token 数: {args.output_tokens}")
    print(f"并发数: {args.concurrency}")
    print(f"请求超时: {args.request_timeout}秒")
    
    # SSE 连接测试
    if not args.skip_sse_test:
        sse_success = asyncio.run(test_sse_connection(args.llm_url, args.api_key, args.model))
        if not sse_success:
            print("\n⚠️  SSE 连接测试失败，但继续进行基准测试...")
            print("   如果测试持续失败，请检查服务器配置和网络连接")
        print()
    
    # 运行测试
    all_results = asyncio.run(run_context_benchmark(
        context_sizes,
        args.num_requests,
        args.output_tokens,
        args.llm_url,
        args.api_key,
        args.model,
        args.request_timeout,
        args.concurrency
    ))
    
    # 创建输出目录
    import os
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"context_benchmark_results_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n详细测试结果已保存至: {results_file}")
    
    # 分析和显示结果
    summary = analyze_context_results(all_results)
    print_context_summary(summary, args.model)
    
    # 保存汇总结果
    summary_file = os.path.join(output_dir, f"context_benchmark_summary_{timestamp}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总结果已保存至: {summary_file}")

if __name__ == "__main__":
    main()