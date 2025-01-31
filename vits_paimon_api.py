import utils
import commons
import torch
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols
import sounddevice as sd

class VITS_PaimonAPI:
	def __init__(self, config_file_path, module_file_path):
		self.config_file_path = config_file_path
		self.module_file_path = module_file_path

		self.hps = utils.get_hparams_from_file(config_file_path)
		self.net_g = SynthesizerTrn(
			len(symbols),
			self.hps.data.filter_length // 2 + 1,
			self.hps.train.segment_size // self.hps.data.hop_length,
			**self.hps.model,
		).cuda()

		self.net_g.eval()

		utils.load_checkpoint(module_file_path, self.net_g, None)
	def get_text(self, text):
		text_norm = text_to_sequence(text,self.hps.data.text_cleaners)
		if self.hps.data.add_blank:
			text_norm = commons.intersperse(text_norm,0)
		text_norm = torch.LongTensor(text_norm)
		return text_norm

	def synthesize_and_play(self, text, length_scale=1.0):
		stn_tst = self.get_text(text)
		with torch.no_grad():
			x_tst = stn_tst.cuda().unsqueeze(0)
			x_tst_lengths=torch.LongTensor([stn_tst.size(0)]).cuda()
			audio = self.net_g.infer(x_tst,x_tst_lengths,noise_scale=.667,noise_scale_w=0.8,length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
			sd.play(audio, self.hps.data.sampling_rate)
			sd.wait()
	

if __name__ == "__main__":
	api = VITS_PaimonAPI("./configs/biaobei_base.json", "./G_1434000.pth")
	api.synthesize_and_play("你好，我是派蒙。")